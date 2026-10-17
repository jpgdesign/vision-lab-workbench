# Sample media attribution

These files retain their individual licenses; the repository's AGPL-3.0 code license does not apply to them.

## COCO images

Ten unmodified JPEGs from COCO 2017 validation. The same image bytes are also included under `../finetuning/coco10_yolo/images/` for a portable training example. Original COCO metadata, exact selection seed, sizes and SHA-256 hashes are preserved in [manifest.json](manifest.json). Additional author/title/current-license lookups are in [flickr_attribution.json](flickr_attribution.json).

| Image | Author | Title / source page | License recorded by COCO | Current Flickr license |
|---|---|---|---|---|
| [000000025424.jpg](images/000000025424.jpg) | James Broad | [Igor Andreev](https://www.flickr.com/photos/kulor/3659789530/) | [BY-NC/2.0](http://creativecommons.org/licenses/by-nc/2.0/) | CC BY-NC 2.0 |
| [000000118594.jpg](images/000000118594.jpg) | Melinda VanLone | [IMG_7233toPost](https://www.flickr.com/photos/pixel97/7766356622/) | [BY-NC/2.0](http://creativecommons.org/licenses/by-nc/2.0/) | CC BY-NC 2.0 |
| [000000173008.jpg](images/000000173008.jpg) | Unavailable in current Flickr response | [Title unavailable](https://www.flickr.com/photo.gne?id=1157171906) | [BY-SA/2.0](http://creativecommons.org/licenses/by-sa/2.0/) | Not verifiable |
| [000000265816.jpg](images/000000265816.jpg) | siska maria eviline | [Fort Santiago](https://www.flickr.com/photos/meequalthecurve/8763417766/) | [BY-NC-ND/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | CC BY-NC-ND 2.0 |
| [000000415727.jpg](images/000000415727.jpg) | Dan RK | [bat_selection](https://www.flickr.com/photos/kueckfamily/7243978106/) | [BY-SA/2.0](http://creativecommons.org/licenses/by-sa/2.0/) | CC BY-SA 2.0 |
| [000000459887.jpg](images/000000459887.jpg) | yummysmellsca | [Freckled Nanners](https://www.flickr.com/photos/reid-bee/5606990845/) | [BY-NC-ND/2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) | CC BY-NC-ND 2.0 |
| [000000493334.jpg](images/000000493334.jpg) | JeffreyPutman | [Birthday Cake](https://www.flickr.com/photos/jeffreyputman/94672746/) | [BY-NC-SA/2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) | CC BY-NC-SA 2.0 |
| [000000495146.jpg](images/000000495146.jpg) | ell brown | [South West Trains - Lymington Harbour - Class 450](https://www.flickr.com/photos/ell-r-brown/9211776644/) | [BY/2.0](http://creativecommons.org/licenses/by/2.0/) | CC BY-NC-SA 2.0 |
| [000000503755.jpg](images/000000503755.jpg) | NAPARAZZI | [ABIGAIL SPEARS](https://www.flickr.com/photos/naparazzi/4286031910/) | [BY-SA/2.0](http://creativecommons.org/licenses/by-sa/2.0/) | CC BY-SA 2.0 |
| [000000527695.jpg](images/000000527695.jpg) | jules:stonesoup | [roast broccoli](https://www.flickr.com/photos/stone-soup/7871176758/) | [BY/2.0](http://creativecommons.org/licenses/by/2.0/) | CC BY 2.0 |

COCO image 173008's original Flickr page is no longer resolvable through the public metadata endpoint; its creator/title cannot currently be verified. We preserve its original source URL and COCO-recorded CC BY-SA 2.0 license instead of inventing attribution. Image 495146 was recorded by COCO as CC BY 2.0, while the current Flickr metadata reports CC BY-NC-SA 2.0; both records are retained.

Several images have NonCommercial, NoDerivatives and/or ShareAlike conditions. This is a collection for noncommercial educational demonstrations, not a uniformly permissive media pack. Preserve these notices and check each image's linked license for your intended reuse, including modified visualizations. No modified image or contact sheet is distributed here. No endorsement by the creators is implied.

## COCO annotations

Annotations and captions are by the **COCO Consortium**, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in the [official terms](https://cocodataset.org/#termsofuse). The JSONs in `annotations/` are subsets selected from the official 2017 annotation archive; YOLO labels under `../finetuning/coco10_yolo/` are coordinate/class-index conversions of these annotations. Category IDs, source annotation IDs, and image licenses remain in the JSONs. The conversion and split are documented in that dataset's `provenance.json`.

## Video

**Freezy Traffic** by [LIFEOFVIDS](https://vimeo.com/lifeofvids), [original Vimeo source](https://vimeo.com/206635019), [Wikimedia Commons file page](https://commons.wikimedia.org/wiki/File:Freezy_Traffic.webm), licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The downloaded WebM is unchanged: no trimming, overlays or transcoding. Its hash, source-page revision and decode details are in [video/source.json](video/source.json). The video shows street traffic in Montréal and has no ground-truth detection or action labels.

The earlier local Pexels clip is not part of this public example collection.
