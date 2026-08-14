# Data and software sources

- **Walk Score points:** `data/toronto_walkscore_extended.csv` contains 796 successful page reads collected on 13 August 2026. `scrape_walkscore.py` records the requested coordinates, returned score, page title, and source URL. Walk Score is a trademark of Redfin. This repository is an independent cartographic study and is not affiliated with Redfin.
- **Toronto boundary:** `data/boundary/` contains the WGS84 regional municipality boundary distributed through the [City of Toronto Open Data portal](https://open.toronto.ca/). The bundled readme and projection file are retained.
- **Road geometry:** `data/gta_major_roads.json` is an Overpass API export from 13 August 2026. The file records its OpenStreetMap attribution and ODbL notice in `osm3s.copyright`. © OpenStreetMap contributors.
- **Basemap tiles:** `prepare_inputs.py` downloads CARTO `light_nolabels` tiles when rebuilding the UV texture. Tiles contain OpenStreetMap data. Follow the [CARTO attribution requirements](https://carto.com/attributions) when redistributing rendered maps.
- **Forge3D:** the verified render used [milos-agathon/forge3d](https://github.com/milos-agathon/forge3d) at commit `f5db54f95d202681f95dad649162d18efdae8987`.

The MIT licence covers the repository's code. Third-party data keeps its original terms.
