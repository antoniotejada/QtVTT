# QtVTT

Qt Virtual Table Top 

Virtual Table Top written in Python and Qt

## Screenshots


### Out of the frying pan into the fire
![image](https://user-images.githubusercontent.com/6446344/205397907-874440f5-d490-4925-bb9b-ff07636287d5.png)

## Video


### Initial line of sight test
https://user-images.githubusercontent.com/6446344/205396105-c2e846ed-1e78-4b48-b261-000bae25c6bb.mp4

[Dungeown Scrawl v2](https://app.dungeonscrawl.com/) map imported from [One Page Dungeon by Watabou](https://watabou.itch.io/one-page-dungeon), with doors as walls, player and DM views, line of sight, and line of sight debugging.

### Multiple tokens, wall & door collisions, walking around opening doors
https://user-images.githubusercontent.com/6446344/205395528-095dee66-6fb8-4a85-9f25-d4a1b818802d.mp4

## Features

- Written in Python and using PyQt, some html for the HTTP serving player's view.

### Main app
- Import [Dungeon Scrawl v1](https://probabletrain.itch.io/dungeon-scrawl) and [v2](https://app.dungeonscrawl.com/) maps with walls and doors
- Load tokens
- Move tokens with mouse or keyboard
- Line of sight display for main token
- Wall and door collision detection for main token
- Door opening/closing with proper line of sight recalculation
- Player view with line of sight, exported via http (see http server)
- Dockable windows for multiple monitor support
- Player view resolution independent from DM resolution
- Ambient music playlist

### HTTP server
- Visualize player's view with line of sight
- Periodic image autorefresh
- Normal/Fullscreen

## Requirements
- Python 2.7
- PyQt5


## Todo 
### Main app
- Import Universal VTT (Dungeondraft)
- Import .mod/.pak (Fantasy grounds)
- Export to Dungeon Scrawl for re-rendering (even re-render under the hood?) 
- Music objects, area, ambient playlist, fading in / out, shuffle, ordered (see https://tabletopaudio.com/)
- Hide/Show objects (secret doors, one way walls, secret areas, hidden tokens, 
  fogged areas, etc)
- Layers, move objects closer/further
- Light sources and illumination/fog of war
- Dimmed out of line of sight but previously visited places (background only,
  with no tokens)
- Edit walls and doors
- Import images (pngs/jpegs...)
- Image/token browser
- Shared pointers and ephemeral drawing 
- Wall, door, token, image, drawing, fog zone, creation/deletion
- Permanent drawing (spell areas, fire splash, game-clock timer areas, etc)
- Measurement tools (cone, circle, line, square)
- Scene saving
- Token states/stats (name, size, hit points, life level, stunned, game-clock timers, etc)
- Campaign/scenes management
- Text editor with markdown editing with cross links/embedded tools (scenes, random tables, monsters, etc)
- Undo/redo
- Record playing/combat history
- Die rolling
- Token designer (import any image, rotate, scale, circle cut and draw ring)
- AD&D2E Encounter builder (calculate hit points, treasure, tokens, drop on scene)
- AD&D2E Monster browser
- AD&D2E PHB/DMG browser
- AD&D2E spell browser
- AD&D2E Combat tracker (initiative, hit points, round counting for game-clock timer effects, etc)
- AD&D2E Treasure generator
- AD&D2E Nested random tables
- AD&D2E character sheets
- AD&D2E character creation
- AD&D2E Party tracker (inventory, weight, treasure splitter...)
- Sharing handouts
- Chat
- Video
- 3D View with miniatures on flat or 3D map (top, ortho)
- Vector Editor (layers, blends, assets, etc) 
- Rotate/pan/scale shared view
- Send shared view via dirty rects

### HTTP Server
- Main page with available player views/handouts/character sheets
- Local pointer/drawing
- Local panning/zooming (won't work for 3D view/will require webgl)
- Local token label/stats drawing for better text clarity
- Local drawing for lighter bandwidth (send background, tokens, fog polygons - may be insecure and incompatible with 3d view)
- Per-player player view
- Character sheet sharing (per player)
- Chat
- Video

