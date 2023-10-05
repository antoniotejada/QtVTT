### Playerviewport synchronized to the player and remote views

![image](https://github.com/antoniotejada/QtVTT/assets/6446344/23c6fad7-4ae9-4b02-871c-ff77d70cf87e)
# QtVTT

Qt Virtual Table Top 

Virtual Table Top written in Python and Qt

## Screenshots


### Out of the frying pan into the fire
![image](https://user-images.githubusercontent.com/6446344/205397907-874440f5-d490-4925-bb9b-ff07636287d5.png)

### Player View, DM View, Browser, and Tree

![image](https://user-images.githubusercontent.com/6446344/205719486-b1cd811e-61eb-4985-a11e-a7fa2fbbe253.png)

### Documentation browser with TOC, term filter and search, circular walls

![image](https://user-images.githubusercontent.com/6446344/206880662-6bb845c3-a5ce-48a9-81bd-bdf7380bb8b1.png)

### Clipped fog polygons, light range, hidden tokens

![image](https://user-images.githubusercontent.com/6446344/207979570-083f0404-b36d-4d6a-b69e-4cd095d546f7.png)

### Documentation browsing, combat tracker, encounter builder

![image](https://user-images.githubusercontent.com/6446344/208770713-77b870b3-4588-45d0-a009-cec9ce1d8307.png)

### Fog of war, display token initiative order

![image](https://user-images.githubusercontent.com/6446344/210019679-b6cf5449-ef91-4268-881d-7705bc1dc030.png)

### Shared and private handouts

![image](https://user-images.githubusercontent.com/6446344/210111727-1c3b2dbb-aa3f-4fe4-9ca3-d237a61b59bc.png)

### Editable walls

![image](https://user-images.githubusercontent.com/6446344/210600981-3d17424b-45d8-49dd-a153-a6ebdde90abf.png)

### Text Editor with outline, search, blockquote, text formats, headings, lists, image, table

![image](https://user-images.githubusercontent.com/6446344/212931148-85ed2f4b-0dfc-460f-8dfb-6a0f5a398e05.png)

### Playerviewport synchronized to the player view, remote view in edge app mode

![image](https://github.com/antoniotejada/QtVTT/assets/6446344/715eb845-a2b9-4c71-b1fb-4c0d979ba534)


## Videos


### Initial line of sight test
https://user-images.githubusercontent.com/6446344/205396105-c2e846ed-1e78-4b48-b261-000bae25c6bb.mp4

[Dungeown Scrawl v2](https://app.dungeonscrawl.com/) map imported from [One Page Dungeon by Watabou](https://watabou.itch.io/one-page-dungeon), with doors as walls, player and DM views, line of sight, and line of sight debugging.

### Multiple tokens, wall & door collisions, walking around opening doors
https://user-images.githubusercontent.com/6446344/205395528-095dee66-6fb8-4a85-9f25-d4a1b818802d.mp4

### Scene creation, token resizing, token labeling, circular walls, documentation browsing, encounter builder, combat tracker

https://user-images.githubusercontent.com/6446344/208662246-48a0f31f-b2f3-45e3-915e-3004dea4d36d.mp4

### Line of sight, fog of war, light range

https://user-images.githubusercontent.com/6446344/209599973-5bc5334e-0cb4-4e40-ab0f-aa14b97c793c.mp4

### Setting up walls fron scratch, editing walls to add openings

https://user-images.githubusercontent.com/6446344/213870729-844c23c7-9998-4427-979b-842faaaab076.mp4

### Automatic wall creation

https://user-images.githubusercontent.com/6446344/215364970-b693fa03-37bf-4a86-a693-935307eb1290.mp4

### Open Dyson Logos page and automatic wall creation

https://user-images.githubusercontent.com/6446344/217021348-e1ba535a-af18-4c77-b87b-6f586a6f1805.mp4


## Features

- Written in Python and using PyQt, some HTML and SVG for the HTTP serving
  player's view.

### Main app
- Import [Dungeon Scrawl v1](https://probabletrain.itch.io/dungeon-scrawl) and [v2](https://app.dungeonscrawl.com/) maps with walls and doors
- Load tokens
- Move tokens with mouse or keyboard
- Line of sight display any token
- Wall and door collision detection for any token
- Door opening/closing with proper line of sight recalculation
- Player view with line of sight, exported via http (see http server)
- Dockable windows for multiple monitor support
- Player view resolution independent from DM resolution
- Per scene music playlist
- HTML Browser
- Scene tree window
- Creating, loading and saving scenes
- Importing images, tokens, music, handouts, texts
- Deleting tokens, images, walls, doors, music, handouts, texts
- Cut, copy, and paste selected tokens
- Select point on single click, full wall on double click (so it can be deleted,
  moved, etc wholesale)
- Editing token labels
- DM fulll and view screenshots
- HTML documentation browser with quick filter, table of contents and search
  result navigation intra and inter documents
- Grid drawing
- Snap to grid
- Clip line of sight to different light ranges (candles, torches, etc) 
- AD&D2E Encounter Builder with difficulty rating and importing into the current
  scene, filter monsters and add them as tokens to the current scene.
- AD&D2E Combat tracker for rolling initiative, hit points, attacks, damages for
  the tokens in the current scene.
- Fog of war, show faded visited areas
- Token tooltips with hit points, AC, THAC0, etc
- Token center label with initiative order
- Shared and private handouts
- Create and edit walls, add points after a given point, remove a point, split
  wall in two, remove empty walls
- Text editor for adventure script, with realtime table of contents, search,
  (some) markdown tag and keyboard shortcut formatting support, tables, and
  images.
- Grid cell size resizing, panning
- Automatic wall creation via contour detection
- Open Dyson Logos page as campaign (eg https://dysonlogos.blog/2022/01/08/heart-of-darkling-frog-tower/ )
- Show player viewport on DM view, synchronize to player's and remote views

### HTTP server
- Visualize player's view with line of sight
- Periodic image autorefresh
- Normal/Fullscreen
- Draggable tokens with labels (just dummy for now, not sync'ed with the server)
- SVG or HTML
- Main page with available player views, handouts
- Serve handouts

## Requirements
- Python 2.7
- PyQt5
- numpy (wall detection disabled otherwise)

## Installation
- Won't work as is yet, will fail to run due to missing (copyrighted) AD&D 2E
  data files, and hardcoded asset (fonts, default token...) and saving directory
  structure.
- Shouldn't be hard to hack to run with few changes, but functionality dependent
  on those AD&D 2E data files will fail (Documentation Browser, Encounter
  Builder, Combat Tracker...)

## Todo 
### Main app
- Add sample scenes
- Don't hard-code documentation/ abstract it out in rulesets/load documentation
  menu
- Import Universal VTT (Dungeondraft)
- Import .mod/.pak (Fantasy grounds)
- Invisible walls: the wall hides the tokens behind but without "fog" (eg to
  make hiding in trees/bushes less obvious)
- Window walls: the wall allows line of sight through but collides like a solid
  wall
- Support animated tokens/images (maps) in mp4, webm, gif, etc (eg see
  https://dynamicdungeons.com/). Will need the http client to receive the map
  and tokens/viz separately (also, record animated videos from DungeonDraft or
  Fantasy Grounds Unity using eg XBOX game bar)
- Use backfacing lines to project the fog, should give a more natural fog for trees, terrain, etc?
- Export to Dungeon Scrawl for re-rendering (even re-render under the hood?) 
- Music objects, area, fading in / out, shuffle, ordered (see https://tabletopaudio.com/, https://www.youtube.com/@bardify)
- Hide/Show objects (secret doors, one way walls, secret areas, fogged areas,
  etc)
- Label objects for text on maps (can work around by using tokens of numbers for
  just numbers or the label of transparent tokens? Would need DM-only tag or
  hide it but without too much fading?)
- Layers, groups, stack objects up/down
- Light sources and illumination/flickering/animated lights
- Edit doors
- Multiple floors support
- Image/token browser
- Shared pointers and ephemeral drawing 
- Wall, door, token, image, drawing, fog zone, creation/deletion
- Permanent drawing (spell areas, fire splash, game-clock timer areas, grids, 
  areas of fog of war, etc)
- Measurement tools (cone, circle, line, square)
- Token states/stats/heading (stunned, game-clock timers, etc)
- Campaign/scenes management
- Encounter management: Store multiple encounters in scenes, edit via encounter
  builder, add encounter to combat tracker, add/drag/copy&paste other tokens to
  combat tracker
- Copy paste of any object
- Cross links/embedded tools in text editor (scenes, random tables, monsters,
  etc)
    - See https://jsigvard.com/dnd/tables.php for random tables
- Markdown text import/edit
- Undo/redo
- Record playing/combat history
- Dice rolling
- Token designer (import any image, rotate, scale, tint, circle cut and draw
  ring)
- AD&D2E spell browser
- AD&D2E Treasure generator
- AD&D2E Nested random tables
- AD&D2E character sheets
- AD&D2E character creation
- AD&D2E Party tracker (inventory, weight, treasure splitter...)
- Record audio of local app (allow transcript? test with bluetooth micro?)
- Record video of local app and/or server UI 
- Chat
- Video
- 3D View with miniatures on flat or 3D map (top, ortho)
- Vector Editor (layers, blends, assets, etc) 
- Send shared view via dirty rects

### HTTP Server
- Local pointer/ephemeral drawing for signaling, etc
- Local distance measuring
- Local panning/zooming (won't work for 3D view/will require webgl)
- Local token label/stats drawing for better text clarity
- Local token roster drawing for initiative visualization, current turn, etc
- Local token moving/dragging (no NPCs unless it's a DM view? no viz updates?
  approve movements? local render of shadow tokens with original positions?)
- Local grid drawing
- Local drawing for lighter bandwidth (send background, tokens, fog polygons -
  may be insecure and incompatible with 3d view)
- Animated maps by pulling the mp4/webm and drawing on top
- Per-player player view
- Send current music (data url or fetched from http server)
- Character sheet sharing (per player)
- Chat
- Video

