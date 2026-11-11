# Tiny Dublin

An isometric pixel-art map of metro Dublin, from Lucan to Howth and from Swords to Killiney Hill: about 650 km² as one 124,928 × 83,968 px image you can zoom from the whole city down to single streets.

**Live map:** https://dqnamo.github.io/tiny-dublin/

It follows the method of [tiny.london](https://tiny.london/how) (Boris Starkov), which itself built on [isometric.nyc](https://github.com/cannoneyed/isometric-nyc) (Andy Coenen).

## How it was made

1. **Render.** Google Photorealistic 3D Tiles are drawn in headless Chromium with three.js and one fixed orthographic camera (tilted 45°, turned 15°, 0.293 m per pixel). Each of the 1,716 shots (2,048 px) is the same camera with its view frustum shifted by whole shots, so the shots tile to the pixel. Shots are rendered in parallel on Modal CPU containers and checked for missing tiles and low detail before they are kept.
2. **Draw.** Each shot area is redrawn by the public [tiny.london LoRA](https://huggingface.co/ktoyame/tiny-london-lora) on Qwen-Image-2.1-Turbo. It runs on Modal H100s as one shared latent per 12,288 px block (MultiDiffusion: 1,024 px windows on a 768 px stride, every other step on a grid shifted 384 px). Each block grows out of its finished neighbours (pinned latents, RePaint), redraws a 384 px strip of them and blends back over 256 px, so there are no seams. 62 blocks run in four phases by the parity of their row and column.
3. **Sea.** Open sea far from the shore is not drawn by the model. It is pinned to a flat water colour from the start, using the OpenStreetMap coastline refined with the render so that sand flats at Sandymount and Bull Island are still drawn.
4. **Clean-up.** As in London, the model sometimes draws Big Ben or the Houses of Parliament on empty water. Calm water (OpenStreetMap water and flat, water-coloured render) is repainted flat. The few landmarks left are patched with the colour of their surroundings, and the map is clipped to the metro box.
5. **Viewer.** The map is a Deep Zoom pyramid of 512 px WebP tiles shown with [OpenSeadragon](https://openseadragon.github.io/), with search over 261 places from OpenStreetMap.

## Layout

```
docs/                 the website (GitHub Pages): viewer, places, tile pyramid
pipeline/pilot/       2 x 2 km pilot round O'Connell Bridge; shared camera maths (geo.py)
pipeline/metro/       the full metro run
  plan.py             shot lattice for the metro box
  masks.py            draw / sea / outside mask from the OSM coastline
  modal_render.py     render shots on Modal (CPU)
  modal_overview.py   low-res summary of the render; used by refine_sea.py
  refine_sea.py       keep sand flats out of the sea fill
  modal_draw.py       block-by-block drawing on Modal (H100); orchestrate() runs all phases
  water_mask.py       where to repaint water; modal_finalize.py applies it and clips
  modal_pyramid.py    Deep Zoom pyramid; reencode_for_pages.py shrinks it for Pages
  make_places.py      place search data
```

To run the pipeline you need a Google Maps Platform key with the Map Tiles API, and a Modal account. Copy `.env.example` to `.env`. Expect Google's daily limit on 3D Tiles root requests to matter: the renderer keeps one page per container and retries only failed tiles to stay well under it.

## Cost

On Modal: about $44 for rendering (including failed and re-rendered passes), about $78 for drawing on H100s, and a few dollars for the rest, so **$127** in all. Google Map Tiles usage is billed separately.

## Credits and licences

- Map imagery is derived from **Google Photorealistic 3D Tiles**. Map data ©2026 Google.
- Style: the **tiny.london LoRA** by Boris Starkov, released under the **Qwen Research License (non-commercial research and evaluation only)**. The base model is Qwen-Image-2.1-Turbo by the Qwen team; built with Qwen. The map inherits this non-commercial restriction.
- Renderer adapted from **isometric-nyc** by Andy Coenen (MIT licence).
- Coastline, water and place names © **OpenStreetMap contributors** (ODbL).
- Method after **tiny.london** and **isometric.nyc**.
