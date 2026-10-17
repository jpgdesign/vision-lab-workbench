const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

export function jobPollingDelay(jobs) {
  return jobs.some(job => job.kind === 'inference' && ['queued', 'running'].includes(job.status)) ? 225 : 2200;
}

function newerJob(current, incoming) {
  if (current && (TERMINAL.has(current.status) || (current.status === 'running' && incoming.status === 'queued'))) return current;
  return incoming;
}

export function mergeJobSnapshot(current, incoming) {
  const previous = new Map(current.map(job => [job.id, job]));
  const received = new Set(incoming.map(job => job.id));
  return [...current.filter(job => !received.has(job.id)), ...incoming.map(job => newerJob(previous.get(job.id), job))];
}

export function mergeJobUpdate(current, incoming) {
  if (!current.some(job => job.id === incoming.id)) return [incoming, ...current];
  return current.map(job => job.id === incoming.id ? newerJob(job, incoming) : job);
}

export class JobPoller {
  constructor({ fetchJobs, getJobs, onJobs, onSuccess = () => {}, onError = () => {},
    setTimer = (callback, delay) => setTimeout(callback, delay), clearTimer = id => clearTimeout(id) }) {
    Object.assign(this, { fetchJobs, getJobs, onJobs, onSuccess, onError, setTimer, clearTimer });
    this.stopped = true;
    this.inFlight = false;
    this.generation = 0;
    this.timer = null;
    this.controller = null;
    this.rerun = false;
  }

  start() {
    if (!this.stopped) return;
    this.stopped = false;
    this.schedule(0);
  }

  schedule(delay) {
    this.clearTimer(this.timer);
    this.timer = this.setTimer(() => { this.timer = null; void this.poll(); }, delay);
  }

  wake() {
    if (this.stopped) return;
    this.generation += 1;
    this.clearTimer(this.timer);
    this.timer = null;
    if (this.inFlight) {
      this.rerun = true;
      this.controller?.abort();
    } else this.schedule(0);
  }

  stop() {
    this.stopped = true;
    this.generation += 1;
    this.clearTimer(this.timer);
    this.timer = null;
    this.controller?.abort();
  }

  async poll() {
    if (this.stopped || this.inFlight) return;
    this.inFlight = true;
    const generation = this.generation;
    const controller = new AbortController();
    this.controller = controller;
    try {
      const jobs = await this.fetchJobs(controller.signal);
      if (!this.stopped && generation === this.generation) {
        this.onJobs(jobs);
        this.onSuccess();
      }
    } catch (error) {
      if (!this.stopped && generation === this.generation && error.name !== 'AbortError') this.onError(error);
    } finally {
      if (this.controller === controller) this.controller = null;
      this.inFlight = false;
      if (!this.stopped) this.schedule(this.rerun ? 0 : jobPollingDelay(this.getJobs()));
      this.rerun = false;
    }
  }
}
