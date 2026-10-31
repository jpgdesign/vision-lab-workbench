import React, { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, Info, Pin, X } from 'lucide-react';
import { trainingModeInfo } from './trainingModeInfo';
import './trainingModeHelp.css';

const detailLabels = { data: '需要的資料', updates: '哪些部分會更新', retention: '原有能力', inference: '推論與成本', useCase: '適合情境' };

export function TrainingModeHelp({ mode = 'standard', note = '' }) {
  const id = useId(), trigger = useRef(null), panel = useRef(null), closeTimer = useRef(null);
  const skipFocusOpen = useRef(false), hovering = useRef(false);
  const [open, setOpen] = useState(false), [pinned, setPinned] = useState(false);
  const [preview, setPreview] = useState(mode), [position, setPosition] = useState(null);
  const info = trainingModeInfo[preview] || trainingModeInfo.standard;

  function clearClose() { window.clearTimeout(closeTimer.current); }
  function show() {
    clearClose();
    if (!open) { setPreview(mode); setPosition(null); setOpen(true); }
  }
  function close(restoreFocus = false) {
    clearClose(); setOpen(false); setPinned(false);
    if (restoreFocus && document.activeElement !== trigger.current) {
      skipFocusOpen.current = true;
      trigger.current?.focus({ preventScroll: true });
    }
  }
  function scheduleClose() {
    clearClose();
    closeTimer.current = window.setTimeout(() => {
      const focused = document.activeElement;
      if (!pinned && !hovering.current && focused !== trigger.current && !panel.current?.contains(focused)) close();
    }, 180);
  }
  function enter() { hovering.current = true; show(); }
  function leave() { hovering.current = false; scheduleClose(); }
  function enterFromKeyboard() {
    show(); setPinned(true);
    window.requestAnimationFrame(() => {
      const choices = panel.current?.querySelector('.training-mode-help-choices');
      (choices?.querySelector('button[aria-pressed="true"]') || choices?.querySelector('button'))?.focus();
    });
  }

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  useEffect(() => {
    if (!open) return undefined;
    function outside(event) {
      if (!trigger.current?.contains(event.target) && !panel.current?.contains(event.target)) close();
    }
    function escape(event) {
      if (event.key === 'Escape') { event.preventDefault(); close(Boolean(panel.current?.contains(document.activeElement))); }
    }
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return undefined;
    function place() {
      if (!trigger.current || !panel.current) return;
      const rect = trigger.current.getBoundingClientRect(), viewport = window.visualViewport;
      const leftEdge = viewport?.offsetLeft || 0, topEdge = viewport?.offsetTop || 0;
      const viewportWidth = viewport?.width || window.innerWidth, viewportHeight = viewport?.height || window.innerHeight;
      const margin = 12, gap = 8, width = Math.min(620, viewportWidth - margin * 2);
      const bounds = { left: leftEdge + margin, right: leftEdge + viewportWidth - margin, top: topEdge + margin, bottom: topEdge + viewportHeight - margin };
      const available = { below: Math.max(0, bounds.bottom - rect.bottom - gap), above: Math.max(0, rect.top - gap - bounds.top), right: Math.max(0, bounds.right - rect.right - gap), left: Math.max(0, rect.left - gap - bounds.left) };
      // Reserve the same space throughout this open session so switching the
      // explanation never moves its preview buttons away from the pointer.
      const reservedHeight = Math.min(640, bounds.bottom - bounds.top);
      let maxHeight = reservedHeight;
      let left = Math.max(bounds.left, Math.min(rect.right - width, bounds.right - width)), top;
      if (reservedHeight <= available.below) top = rect.bottom + gap;
      else if (reservedHeight <= available.above) top = rect.top - reservedHeight - gap;
      else if (Math.max(available.right, available.left) >= width) {
        left = available.right >= width ? rect.right + gap : rect.left - width - gap;
        top = Math.max(bounds.top, Math.min(rect.top, bounds.bottom - maxHeight));
      } else if (available.below >= available.above) {
        maxHeight = Math.min(maxHeight, available.below);
        top = rect.bottom + gap;
      } else {
        maxHeight = Math.min(maxHeight, available.above);
        top = rect.top - gap - maxHeight;
      }
      // When neither side fits, scroll the card in the larger vertical space;
      // never clamp a full-height card across its own trigger.
      top = Math.max(bounds.top, Math.min(top, bounds.bottom - maxHeight));
      const next = { width, maxHeight, left, top };
      setPosition(previous => previous && Object.keys(next).every(key => previous[key] === next[key]) ? previous : next);
    }
    place();
    const observer = new ResizeObserver(place);
    observer.observe(panel.current);
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    window.visualViewport?.addEventListener('resize', place);
    window.visualViewport?.addEventListener('scroll', place);
    return () => {
      observer.disconnect(); window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true);
      window.visualViewport?.removeEventListener('resize', place); window.visualViewport?.removeEventListener('scroll', place);
    };
  }, [open, preview, note]);

  return <>
    <button ref={trigger} type="button" className="training-mode-help-trigger" aria-expanded={open} aria-controls={open ? id : undefined} aria-haspopup="dialog"
      onMouseEnter={enter} onMouseLeave={leave} onBlur={scheduleClose}
      onFocus={() => { if (skipFocusOpen.current) skipFocusOpen.current = false; else show(); }}
      onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); enterFromKeyboard(); } }}
      onClick={event => { event.preventDefault(); if (event.detail === 0) enterFromKeyboard(); else if (pinned) close(); else { show(); setPinned(true); } }}>
      <Info size={14} aria-hidden="true"/>方式差異
    </button>
    {open && createPortal(<aside ref={panel} id={id} className="training-mode-help-card" role="dialog" aria-modal="false" aria-labelledby={`${id}-title`}
      style={position ? { ...position, visibility: 'visible' } : { visibility: 'hidden' }}
      onMouseEnter={() => { hovering.current = true; clearClose(); }} onMouseLeave={leave} onFocus={clearClose} onBlur={scheduleClose}>
      <header className="training-mode-help-header"><div><h3 id={`${id}-title`}>類別學習方式差異</h3><p>僅查看說明，不變更訓練設定</p></div>
        <div className="training-mode-help-actions">{pinned && <span><Pin size={12} aria-hidden="true"/>已固定</span>}<button type="button" aria-label="關閉方式說明" onClick={() => close(true)}><X size={17} aria-hidden="true"/></button></div>
      </header>
      <div className="training-mode-help-choices" role="group" aria-label="預覽各種訓練方式的說明">
        {Object.entries(trainingModeInfo).map(([value, item]) => <button type="button" key={value} className={preview === value ? 'active' : ''} aria-pressed={preview === value} aria-controls={`${id}-detail`}
          onMouseMove={() => setPreview(value)} onFocus={() => setPreview(value)} onClick={() => { setPreview(value); setPinned(true); }}>
          <span>{item.shortTitle}</span>{value === mode && <small><Check size={11} aria-hidden="true"/>目前選用</small>}
        </button>)}
      </div>
      <section className="training-mode-help-detail" id={`${id}-detail`} aria-labelledby={`${id}-detail-title`}>
        <h4 id={`${id}-detail-title`}>{info.title}</h4><p className="training-mode-help-summary">{info.summary}</p>
        <dl>{Object.entries(detailLabels).map(([key, label]) => <div key={key}><dt>{label}</dt><dd>{info[key]}</dd></div>)}</dl>
      </section>
      {note && <p className="training-mode-help-note"><Info size={14} aria-hidden="true"/><span>{note}</span></p>}
      <footer>懸停或聚焦選項可比較說明；點按可固定資訊卡，Esc 可關閉。</footer>
    </aside>, document.body)}
  </>;
}
