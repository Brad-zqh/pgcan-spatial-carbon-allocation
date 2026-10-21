# Released data and provenance

`results/` contains actual author-generated model predictions, constructed
targets and aggregate experimental results, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). There are 1,980 anonymous
cells, 31 evaluated configurations and 405 outer-fold/seed records. These
are **proxy reconstruction results, not measured CO2 emissions**.

Cell order is identical across prediction arrays and `cell_targets.csv`.
`cell_id` is a fresh opaque row identifier. Geographic coordinates, POI IDs,
enterprise names, site addresses and source records are not published.
Missing raw predictors mean this release supports exact metric recomputation,
but full model retraining requires separately authorized predictor inputs.

| Material | Release decision |
|---|---|
| Author code, equations, fixed configs, splits, aggregate metrics | Included |
| Anonymous OOF predictions and constructed targets | Included; actual values, no synthetic substitutions |
| Commercial POI records, linked entity lists, patents with unclear source terms | Not redistributed; schema and local import workflow provided |
| Esri World Imagery tiles/cache | Not redistributed; provider access terms apply |
| OSM vector database/extracts | Not bundled; ODbL attribution/share-alike obligations apply if obtained separately |
| User-supplied China administrative boundaries | Not bundled; external provenance requires separate permission |
| Natural Earth background | Public domain; obtain from the provider |
| Cited papers and third-party reference implementations | Not redistributed |

Primary sources checked on 2026-10-22:
- https://www.openstreetmap.org/copyright
- https://www.naturalearthdata.com/about/terms-of-use/
- https://www.esri.com/en-us/legal/terms/web-site-service
- https://support.esri.com/en-us/knowledge-base/what-is-the-correct-way-to-cite-an-arcgis-online-basema-000012040

An accessible website or a research-use copy does not establish a blanket
right to publish its underlying data. This repository applies no MIT license
to third-party data. Store authorized local inputs only under ignored
`data/local/`; never commit tokens, browser cookies or account credentials.
