#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

"""
Qt Virtual Table Top
(c) Antonio Tejada 2022

"""

import io
import json
import logging
import math
import os
import re
import SimpleHTTPServer
import SocketServer
import sys
import thread
import zipfile


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *


class LineHandler(logging.StreamHandler):
    def __init__(self):
        super(LineHandler, self).__init__()

    def emit(self, record):
        text = record.getMessage()
        messages = text.split('\n')
        indent = ""
        for message in messages:
            r = record
            r.msg = "%s%s" % (indent, message)
            r.args = None
            super(LineHandler, self).emit(r)
            indent = "    " 


def setup_logger(logger):
    """
    Setup the logger with a line break handler
    """
    logging_format = "%(asctime).23s %(levelname)s:%(filename)s(%(lineno)d):[%(thread)d] %(funcName)s: %(message)s"

    logger_handler = LineHandler()
    logger_handler.setFormatter(logging.Formatter(logging_format))
    logger.addHandler(logger_handler) 

    return logger

logger = logging.getLogger(__name__)
setup_logger(logger)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)

def report_versions():
    logger.info("Python version: %s", sys.version)

    # Numpy is only needed to apply gamma correction
    np_version = "Not installed"
    try:
        import numpy as np
        np_version = np.__version__
        
    except:
        warn("numpy not installed, image filters disabled")
    logger.info("Numpy version: %s", np_version)
    

    logger.info("Qt version: %s", QT_VERSION_STR)
    logger.info("PyQt version: %s", PYQT_VERSION_STR)

    pyqt5_sqlite_version = "Not installed"
    pyqt5_sqlite_compile_options = []
    try:
        from PyQt5.QtSql import QSqlDatabase
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.open()
        query = db.exec_("SELECT sqlite_version();")
        query.first()
        pyqt5_sqlite_version = query.value(0)

        query = db.exec_("PRAGMA compile_options;")
        while (query.next()):
            pyqt5_sqlite_compile_options.append(query.value(0))
        db.close()
    
    except:
        # On Linux QtSql import is known to fail when python-pyqt5.qtsql is not
        # installed, needs 
        #   apt install python-pyqt5.qtsql 
        pass
        
    logger.info("QSQLITE version: %s", pyqt5_sqlite_version)
    logger.info("QSQLITE compile options: %s", pyqt5_sqlite_compile_options)
    logger.info("Qt plugin path: %s", os.environ.get("QT_PLUGIN_PATH", "Not set"))
    logger.info("QCoreApplication.libraryPaths: %s", QCoreApplication.libraryPaths())
    logger.info("QLibraryInfo.PrefixPath: %s", QLibraryInfo.location(QLibraryInfo.PrefixPath))
    logger.info("QLibraryInfo.PluginsPath: %s", QLibraryInfo.location(QLibraryInfo.PluginsPath))
    logger.info("QLibraryInfo.LibrariesPath: %s", QLibraryInfo.location(QLibraryInfo.LibrariesPath))
    logger.info("QLibraryInfo.LibrarieExecutablesPath: %s", QLibraryInfo.location(QLibraryInfo.LibraryExecutablesPath))
    logger.info("QLibraryInfo.BinariesPath: %s", QLibraryInfo.location(QLibraryInfo.BinariesPath))


class Struct:
    "A structure that can have any fields defined."
    def __init__(self, **entries): self.__dict__.update(entries)


def import_png(scene, filepath, size_in_cells, offset_in_cells=(0,0)):
    map_cell_diameter = scene.cell_diameter

    # XXX May want to downscale the image, but random scaling has precision
    #     issues and needs variables to change to float
    qim = QImage(filepath)
    
    img_size_in_pixels = (qim.width(), qim.height())
    logger.debug("img size in pixels %s", img_size_in_pixels)
    
    # XXX Some dungeon scrawl v1 png files are saved with the wrong pixel
    #     height, do rounding and don't assert
    img_cell_size_in_pixels = round(img_size_in_pixels[0] / size_in_cells[0])
    ##assert img_cell_size_in_pixels == (img_size_in_pixels[1] / img_size_in_cells[1])
    ##assert float(img_size_in_pixels[0] / img_size_in_cells[0]) == img_cell_size_in_pixels
    logger.debug("img cell size in pixels %s", img_cell_size_in_pixels)

    img_pixels_per_unit = img_cell_size_in_pixels / map_cell_diameter
    logger.debug("img pixels per unit %s", img_pixels_per_unit)

    # XXX This needs user input at load time or allow panning of the image
    #     after loading
    img_offset_in_units = (offset_in_cells[0] * map_cell_diameter, offset_in_cells[1] * map_cell_diameter)
    logger.debug("img offset in units %s", img_offset_in_units)

    scene.map_image = qim
    scene.img_pixels_per_unit = img_pixels_per_unit
    scene.img_offset_in_units = img_offset_in_units

    return scene
    

def import_ds(scene, filepath):
    # Load a dungeonscrawl .ds data file 
    # See https://www.dungeonscrawl.com/
    
    # Recent ds files are zip compressed with a "map" file inside, try to
    # uncompress first, then parse
    try:
        f = zipfile.ZipFile(filepath)
        # The zip has one member with the name "map"
        f = f.open("map")
        #f.seek(0, os.SEEK_SET)

    except:
        logger.warning("ds file triggered exception as zip, retrying as plain json")
        f = open(filepath, "r")
        
    js = json.load(f)
    f.close()


    # There are two versions of dungeonscrawl each with different data file layouts
    # v1 https://probabletrain.itch.io/dungeon-scrawl
    # v2 https://app.dungeonscrawl.com/
    ds_version = 2 if "state" in js else 1

    # In Dungeonscrawl there are three units of measurement:
    # - cells: the name of the png map file contains the width and height of the
    #   map in cells
    # - pixels: the png map file width and height
    # - map units: units used in the geometries inside the map datafile

    # The map datafile "cellDiameter" or "gridCellSize" entry specifies how many
    # map units there are per cell

    # In addition, the png map file is a clipped version of the map, the
    # dimensions in cells are in the png filename but the offset is not and has
    # to be provided manually after the fact
        
    if (ds_version == 1):
        map_cell_diameter = float(js["layerController"]["config"]["gridCellSize"])

    else:
        map_cell_diameter = float(js["state"]["document"]["nodes"]["intial-page"]["grid"]["cellDiameter"])
        
    logger.info("ds version %d map cell diameter %s", ds_version, map_cell_diameter)

    map_walls = []
    map_doors = []
    door_geom_ids = set()

    dungeon_geom_ids = set()

    if (ds_version != 1):
        # Find the door geom ids to process tell doors vs. walls later
        logger.debug("collecting door and dungeon geom ids")
        for node in js["state"]["document"]["nodes"].itervalues():
            geom_id = node.get("geometryId", None)
            if ((geom_id is not None) and (node.get("name", "").startswith("Door"))):
                logger.debug("Found door geom_id %s", geom_id)
                door_geom_ids.add(geom_id)

            if ((geom_id is not None) and (node.get("name", "").startswith("Dungeon layer"))):
                logger.debug("Found dungeon geom_id %s", geom_id)
                dungeon_geom_ids.add(geom_id)
    
    
    logger.debug("collecting walls and doors")
    if (ds_version == 1):
        layers = js["layerController"]["layers"]

    else:
        layers = js["data"]["geometry"]
        

    # Maximum coalescing distance squared
    # Any two wall segments shorter than this will be coalesced in a single one
    # removing the middle point
    # Using less than cell/8.0 as distance can wrongly coalesce part of a door
    # causing light leaks on v2, don't coalesce too aggressively and special-case
    # doors
    # XXX Make this an import time option? Have the user manually tweak it
    #     afterwards? disable coalescing for doors, but what about stairs?
    if (ds_version == 1):
        # On v1 16.0 coalescing is known to cause deformations in the geometry
        # close to doors, not clear it's an issue but use 32.0 to be on the safe
        # side
        max_coalesce_dist2 = (map_cell_diameter/32.0) ** 2
    else:
        max_coalesce_dist2 = (map_cell_diameter/8.0) ** 2
    duplicates = set()
    coalesced_walls = 0
    duplicated_walls = 0
    coalesce_walls = True
    for layer in layers:
        if (ds_version == 1):
            # Doors on v1 are three squares (so 24 points) but also lots of 
            # replicated points at the intersections:
            # - 10 left side between top and mid 
            # - 10 left side at top
            # - ...
            # all in all, 45 points followed by 25 followed by 25
            # 25 are the door hinges, 45 is the door pane
            # XXX Open doors on v1 don't let line of sight go through, looks like
            #     there's extra geometry that prevents it, investigate (or remove
            #     manually once map is editable)
            is_door = False
            shapes = layer.get("shape", {}).get("shapeMemory", [[]])
            # Looks like the shape memory is some kind of snapshots/undo
            # history, only look at the latest version
            currentShapeIndex = layer.get("shape", {}).get("currentShapeIndex", 0)
            shapes = shapes[currentShapeIndex]
            lines = []

        else:
            # XXX Dungeonscrawl doesn't support secret/hidden doors, but they
            #     could be done by 
            #     - using doors with alpha blending/special color/stroke/width
            #     - placing a door over an existing wall and detect at import time
            #     - placing some image over an existing wall
            
            #     Care would need to be taken so the doors are not obvious in
            #     the png map, eg by using translucent door over a wall (but
            #     that will place a wall that will need to be removed manually
            #     or detected at import time or allow movement/line of sight
            #     through that door once opened)

            #     Not clear how it would work for trapdoors, etc, probably needs
            #     to key on a trapdoor image, etc. An image
            is_door = (layer in door_geom_ids)
            if ((not is_door) and (layer not in dungeon_geom_ids)):
                # XXX Restricting to a layer called "Dungeon layer..." removes
                #     some non-walls (eg lakes) but can also be too conservative
                #     when other dungeon layers exist or have been renamed?
                #     Would probably need an option or grouping different layers
                #     on different groups so they can be deleted wholesale after
                #     the fact?
                continue
            layer = layers[layer]
            shapes = layer.get("polygons", [])
            # Nest shape so v1 and v2 parse the same way
            # XXX Should flatten v1 instead?
            #shapes = [[shape] for shape in shapes]
            lines = layer.get("polylines", [[]])
        
        for shape1 in shapes:
            for shape2 in shape1:
                door = []
                x0, y0 = None, None

                if (ds_version == 1):
                    # 25 is the door hinges, 45 is the door pane
                    is_door = (len(shape2) in [45])
                
                for v in shape2:
                    x1, y1 = v

                    if (x0 is not None):
                        # Coalesce walls that are too small, this fixes performance
                        # issues when walls come from free form wall tool with lots
                        # of very small segments
                        # Don't coalesce doors as they may have small features
                        # on v1
                        if (coalesce_walls and (not is_door) and (
                            # This edge is small
                            ((x1 - x0) ** 2 + (y1 - y0) ** 2 < max_coalesce_dist2) and
                            # The distance between the start and the end is less than min
                            (((x1 - start[0]) ** 2) + ((y1 - start[1]) ** 2) < max_coalesce_dist2)
                        )):
                            logger.debug("Coalescing %s to %s", (x1, y1), start)
                            coalesced_walls += 1

                        else:
                            x0, y0 = start
                            wall = (x0, y0, x1, y1)
                            # Some maps come with thousands of duplicates
                            # XXX Check for duplicates higher up to discard earlier?
                            #     (watch for interactions with coalescing?)
                            if (wall not in duplicates):
                                if (is_door):
                                    door.extend(wall)
                                else:
                                    map_walls.append(wall)

                                duplicates.add(wall)

                            else:
                                logger.debug("Ignoring duplicated %s", wall)
                                duplicated_walls += 1
                            
                            start = x1, y1

                    else:
                        start = x1, y1
            
                    x0, y0 = x1, y1

                if (is_door and (len(door) > 0)):
                    map_doors.append(door)
                
        # These correspond to the door handles, adding as walls
        # XXX Do a better parse that doesn't assume two lines per entry?
        map_walls.extend([line[0] + line[1] for line in lines])

    logger.info("Found %d doors %d valid walls, %d coalesced, %d duplicated", len(map_doors), len(map_walls), 
        coalesced_walls, duplicated_walls)
    
    scene.cell_diameter = map_cell_diameter
    scene.map_walls = map_walls
    scene.map_doors = map_doors

    return scene


def load_scene(ds_filepath, map_filepath, img_offset_in_cells):

    scene = Struct()
    import_ds(scene, ds_filepath)

    # Size in cells is embedded in the filename 
    m = re.match(r".*\D+(\d+)x(\d+)\D+", map_filepath)
    img_size_in_cells = (int(m.group(1)), int(m.group(2)))
    logger.debug("img size in cells %s", img_size_in_cells)

    import_png(scene, map_filepath, img_size_in_cells, img_offset_in_cells)

    return scene

    
# Image bytes (eg PNG or JPEG) between the app thread and the http server thread
# XXX This needs to be more flexible once there are multiple images shared, etc
img_bytes = None
# Encoding to PNG takes 4x than JPEG, use JPEG (part of http and app handshake
# configuration)
# XXX For consistency the html and the path should use image.jpg instead of .png
#     when encoding to jpeg (using the wrong extension is ok, though, because
#     the extension is ignored when a mime type is passed)
#imctype = "image/png"
imctype = "image/jpeg"
imformat = "PNG" if imctype.endswith("png") else "JPEG"
class VTTHTTPRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        if (self.path.startswith("/image.png")):
            logger.debug("get start %s", self.path)

            # XXX This should use the global image or a copy rather than recreating
            #     it here

            logger.debug("wrapping in bytes")
            f = io.BytesIO(img_bytes)
            clength = len(img_bytes)

            logger.debug("returning")
            ctype = imctype

        elif (self.path == "/index.html"):
            ctype = "text/html"
            clength = os.path.getsize("index.html")

            f = open("index.html", "rb")

        else:
            return super(VTTHTTPRequestHandler, self).do_GET(self)

        self.send_response(200)
        self.send_header("Content-Length", str(clength))
        self.send_header("Content-Type", ctype)
        # XXX Only needed for dynamic content, but useful for all when developing
        self.send_header("Cache-Control", "no-cache")
        #self.send_header("Cache-Control", "must-revalidate")

        self.end_headers()        

        chunk_size = 2 ** 20
        chunk = None
        while (chunk != ""):
            chunk = f.read(chunk_size)
            self.wfile.write(chunk)
            
        f.close()
        
    def do_HEAD(self):
        logger.info("head for %r", self.path)
        return super(VTTHTTPRequestHandler, self).do_HEAD()


def server_thread(arg):
    server_address = ("", 8000)
    handler_class = VTTHTTPRequestHandler
    httpd = SocketServer.TCPServer(tuple(server_address), handler_class)

    # XXX In case some computation is needed between requests, this can also do
    #     httpd.handle_request in a loop
    httpd.serve_forever()
    
g_draw_walls = True
g_blend_map_fog = False
g_draw_map_fog = False

class ImageWidget(QScrollArea):
    """
    See https://code.qt.io/cgit/qt/qtbase.git/tree/examples/widgets/widgets/imageviewer/imageviewer.cpp
    
    XXX Note that approach uses a lot of memory because zooming creates a background
        pixmap of the zoomed size. Use a GraphicsView instead? 
        implement scrollbar event handling and do viewport-only zooming?
    """
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)

        self.scale = 1.0
        # XXX This could have three states, fit to largest, fit to smallest and
        #     no fit
        self.fitToWindow = True

        imageLabel = QLabel()
        imageLabel.setBackgroundRole(QPalette.Base)
        imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        imageLabel.setScaledContents(True)
        self.imageLabel = imageLabel

        self.setBackgroundRole(QPalette.Dark)
        self.setWidget(imageLabel)
        self.setWidgetResizable(self.fitToWindow)
        self.setAlignment(Qt.AlignCenter)
        #self.setStyleSheet("background-color:rgb(196, 196, 196);")

        self.viewport().installEventFilter(self)

        self.setCursor(Qt.OpenHandCursor)

    def setImage(image):
        if (image.colorSpace().isValid()):
            image.convertToColorSpace(QColorSpace.SRgb)
        self.setPixmap(QPixmap.fromImage(image))

    def setPixmap(self, pixmap):
        logger.info("setPixmap %s", pixmap.size())
        self.imageLabel.setPixmap(pixmap)

        
    def zoomImage(self, zoomFactor, anchor = QPointF(0.5, 0.5)):
        logger.debug("zoomImage %s %s", zoomFactor, anchor)
        self.scale *= zoomFactor
        self.resizeImage(self.scale * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.horizontalScrollBar(), zoomFactor, anchor.x())
        self.adjustScrollBar(self.verticalScrollBar(), zoomFactor, anchor.y())

    def adjustScrollBar(self, scrollBar, factor, anchor):
        logger.debug("adjustScrollBar anchor %s value %s page %s min,max %s", anchor, scrollBar.value(), scrollBar.pageStep(), (scrollBar.minimum(), scrollBar.maximum()))

        # anchor bottom
        #scrollBar.setValue(int(scrollBar.value() * factor - scrollBar.pageStep() + factor * scrollBar.pageStep()))
        # anchor top
        #scrollBar.setValue(int(scrollBar.value() * factor))
        # aanchor linear interpolation wrt anchor:
        scrollBar.setValue(int(scrollBar.value() * factor + anchor * (factor * scrollBar.pageStep() - scrollBar.pageStep())))
        


    def setFitToWindow(self, fitToWindow):
        logger.info("setFitToWindow %s", fitToWindow)
        self.fitToWindow = fitToWindow
        self.setWidgetResizable(self.fitToWindow)
        if (self.fitToWindow):
            self.scale = 1.0
            self.resizeImage(self.size())

        else:
            QWIDGETSIZE_MAX = ((1 << 24) - 1)
            self.scale = float(self.imageLabel.width()) / self.imageLabel.pixmap().width()
            self.imageLabel.setFixedSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)


    def resizeImage(self, size):
        logger.info("resizeImage %s fitToWindow %s", size, self.fitToWindow)
        pixSize = self.imageLabel.pixmap().size()
        aspectRatio = float(pixSize.width()) / pixSize.height()
        if ((float(size.width()) / size.height()) > aspectRatio):
            dim = size.height()

        else:
            dim = size.width() / aspectRatio

        tweakedSize = QSize(int(dim * aspectRatio), int(dim))

        logger.debug("Resizing to %s", tweakedSize)
        self.imageLabel.setFixedSize(tweakedSize)
        

    def resizeEvent(self, event):
        logger.info("resizeEvent %s", event.size())

        if (self.fitToWindow):
            newSize = event.size()
            self.resizeImage(newSize)
        
        super(ImageWidget, self).resizeEvent(event)

    def mousePressEvent(self, event):
        logger.info("mousePressEvent %s", event.pos())

        if (event.button() == Qt.LeftButton):
            self.setCursor(Qt.ClosedHandCursor)
            self.prevDrag = event.pos()

        else:
            super(ImageWidget, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        logger.info("mouseRelease %s", event.pos())

        if (event.button() == Qt.LeftButton):
            self.setCursor(Qt.OpenHandCursor)
        
        else:
            super(ImageWidget, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        logger.debug("mouseMoveEvent %s", event.pos())

        if (event.buttons() == Qt.LeftButton):
            delta = (event.pos() - self.prevDrag)
            logger.debug("Scrolling by %s", delta)
            
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
                
            self.prevDrag = event.pos()

        else:

            return super(ImageWidget, self).mouseMoveEvent(event)

    def eventFilter(self, source, event):
        logger.debug("ImageWidget eventFilter %s %s", source, event.type())
        # Even if the filter was installed on the viewport, it will receive
        # messages from scrollbars and the label, discard those
        if ((event.type() == QEvent.Wheel) and (source == self.viewport())):
            # XXX This should center on the mouse position
            logger.debug("Wheel 0x%x pos %s", event.modifiers(), event.pos())
            if (self.fitToWindow):
                self.setFitToWindow(not self.fitToWindow)
            zoomFactor = 1.0015 ** event.angleDelta().y() 
            anchor = QPointF(
                event.pos().x() * 1.0 / source.width(), 
                event.pos().y() * 1.0 / source.height()
            )
            self.zoomImage(zoomFactor, anchor)
            
            return True

        return False


class VTTMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(VTTMainWindow, self).__init__(parent)

        ds_filepath = "hidden hold.dsz"
        map_filepath = "hidden hold_42x29.png"
        img_offset_in_cells = (14, 13)

        #ds_filepath = "map.ds"
        #map_filepath = "map_36x25.png"
        #img_offset_in_cells = (-2, -4)
        
        #ds_filepath = os.path.join("fonda.ds")
        #map_filepath = os.path.join("fonda_31x25.png")
        #img_offset_in_cells = (-8, -0)


        #ds_filepath = "Prison of Sky.ds"
        #map_filepath = "Prison of Sky_56x39.png"
        #img_offset_in_cells = (27, 35)

        #ds_filepath = "Monastery of the Leper General.ds"
        #map_filepath = "Monastery of the Leper General_33x44.png"
        #img_offset_in_cells = (8, 27)

        #ds_filepath = "waterv2.ds"
        #map_filepath = "waterv2_67x51.png"
        #img_offset_in_cells = (3, 15)

        #ds_filepath = "map_27x19.ds"
        #ds_filepath = "dungeon (5).ds"
        #map_filepath = "map_27x19.png"
        #img_offset_in_cells = (0, -2)

        #ds_filepath = "secrets.ds"
        #map_filepath = "secrets_47x40.png"
        #img_offset_in_cells = (-2, 11)

        #ds_filepath = "Straight Forest Road Battle Map.ds"
        #map_filepath = os.path.join("maps","Straight Forest Road Battle Map 22x30.png")
        #img_offset_in_cells = (-2, 11)

        ds_filepath = os.path.join("_out", ds_filepath)
        map_filepath = os.path.join("_out", map_filepath)

        if (len(sys.argv) == 4):
            ds_filepath = sys.argv[1]
            map_filepath = sys.argv[2]
            img_offset_in_cells = [int(offset) for offset in sys.argv[3].split(",")]

        scene = load_scene(ds_filepath, map_filepath, img_offset_in_cells)
        self.scene = scene

        self.player = QMediaPlayer()
        self.player.setVolume(50)
        self.playlist = QMediaPlaylist()
        self.player.setPlaylist(self.playlist)

        music_dir = os.path.join("_out", "music")
        for filename in os.listdir(music_dir):
            if (filename.endswith(".mp3")):
                filepath = os.path.join(music_dir, filename)
                logger.debug("Adding media %r", filepath)
                self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
        
            
        bar = self.menuBar()
        file = bar.addMenu("File")
        file.addAction("New")
        file.addAction("save")
        file.addAction("quit")
            
        self.items = [QDockWidget("Player View", self), QDockWidget("DM View", self)]

        imageWidget = ImageWidget()
        # Allow the widget to receive mouse events
        imageWidget.setMouseTracking(True)
        # Allow the widget to receive keyboard events
        imageWidget.setFocusPolicy(Qt.StrongFocus)
        imageWidget.installEventFilter(self)
        self.imageWidget = imageWidget


        gscene = QGraphicsScene()
        # XXX It's not clear the BSP is helping on dynamic scenes with fog
        #     (although the fog is not selectable so it shouldn't be put in the
        #     bsp?)
        ##gscene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.gscene = gscene
        self.map_fog_item = None
        self.map_token_item = None
        self.open_door_items = set()
        self.door_items = set()
        self.wall_items = set()
        self.token_coords = (0, 0)
        self.populateGraphicsScene(gscene, scene, self.token_coords)
        gscene.installEventFilter(self)
        
        view = QGraphicsView(gscene)
        view.installEventFilter(self)
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView = view
            
        self.items[0].setWidget(self.imageWidget)
        self.items[0].setFloating(False)
        self.items[0].setAllowedAreas(Qt.AllDockWidgetAreas)
        self.items[1].setWidget(view)
        self.items[1].setFloating(False)
        self.items[1].setAllowedAreas(Qt.AllDockWidgetAreas)

        # XXX Note Qt QTextEdit supports markdown format since 5.14 but the
        #     installed version is 5.3.1. Would need to roll own markdown to
        #     rich text editor (see
        #     https://doc.qt.io/qt-6/richtext-structure.html for tables, etc) or
        #     to html (note plain markdown doesn't support tables, so ideally
        #     would need to be a mixed markdown/html editor?)
        #     Could also use a webview with markdown.js/marked.js or such
        #     Could also use some python markdown package to translate to richtext
        #     or html
        self.setCentralWidget(QTextEdit())
        self.addDockWidget(Qt.TopDockWidgetArea, self.items[0])
        self.addDockWidget(Qt.TopDockWidgetArea, self.items[1])
        #self.centralWidget().hide()
        self.setWindowTitle("QtVTT")

        # Perform the initial rendering and initialize the imagewidget contents
        self.updateImage()


    def populateGraphicsScene(self, gscene, scene, token_coords):
        gscene.setSceneRect(-scene.img_offset_in_units[0], -scene.img_offset_in_units[1], scene.map_image.width() / scene.img_pixels_per_unit, 
            scene.map_image.height() / scene.img_pixels_per_unit)

        pix = QGraphicsPixmapItem(QPixmap(scene.map_image))
        pix.setPos(-scene.img_offset_in_units[0], -scene.img_offset_in_units[1])
        logger.debug("setting scale to %s", scene.img_pixels_per_unit)
        pix.setScale(1.0 / scene.img_pixels_per_unit)
        gscene.addItem(pix)
        
        use_image = True
        for filename in ["Hobgoblin.png", "Hobgoblin.png", "Goblin.png" ,"Goblin.png", "Goblin.png", "Gnoll.png", "Ogre.png", "Ancient Red Dragon.png", "Knight.png", "Priest.png", "Mage.png"]:
            if (use_image):
                # Note this uses nested items instead of groups since groups
                # change the token rect to include the label but the parent
                # item doesn't contain the child bounding rect
                filepath = os.path.join("_out", "tokens", filename)
                pix = QPixmap(filepath)
                max_token_size = QSize(64, 64)
                # Big tokens are noticeably slower to render, use a max size
                logger.debug("Loading and resizing token %r from %s to %s", filepath, pix.size(), max_token_size)
                pix = pix.scaled(max_token_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pixItem = QGraphicsPixmapItem(pix)

                # Scale to cell size
                pix_scale = scene.cell_diameter * 1.0/pixItem.boundingRect().width()
                # XXX This should check the monster size or such
                if ("Dragon" in filename):
                    pix_scale *= 4.0
                elif ("Ogre" in filename):
                    pix_scale *= 1.5

                pixItem.setScale(pix_scale)

                txtItem = QGraphicsTextItem(pixItem)
                # Use HTML since it allows setting the background color
                txtItem.setHtml("<div style='background:rgb(255, 255, 255);'>%s</div>" % os.path.splitext(filename)[0])
                # Keep the label always at the same size disregarding the token
                # size, because the label is a child of the pixitem it gets
                # affected by it
                # Reduce the font a bit
                font = txtItem.font()
                font.setPointSize(txtItem.font().pointSize() *0.75)
                txtItem.setFont(font)
                txtItem.setScale(1.0/pix_scale)
                # Calculate the position taking into account the text item
                # reverse scale                
                pos = QPointF(
                    pixItem.boundingRect().width() / 2.0 - txtItem.boundingRect().width() / (2.0 * pix_scale), 
                    pixItem.boundingRect().height() - txtItem.boundingRect().height() / (3.0 * pix_scale)
                )
                txtItem.setPos(pos)

                
                item = pixItem

            else:
                brush = QBrush(Qt.black)
                x0, y0 = token_coords
                x1, y1 = x0 + scene.cell_diameter, y0 + scene.cell_diameter
                item = QGraphicsEllipseItem(x0, y0, x1 - x0, y1 -y0)
                item.setBrush(brush)
            item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
            item.setCursor(Qt.SizeAllCursor)
            gscene.addItem(item)
        
        self.map_token_item = item

        pen = QPen(Qt.cyan)
        self.wall_items = set()
        self.all_walls_item = QGraphicsItemGroup()
        logger.debug("Creating %d wall items", len(scene.map_walls))
        for wall in scene.map_walls:
            item = QGraphicsLineItem(*wall)
            # Items cannot be selected, moved or focused while inside a group,
            # the group can be selected and focused but not moved
            item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable)
            item.setPen(pen)
            item.setData(0, wall)
            self.wall_items.add(item)
            self.all_walls_item.addToGroup(item)

        gscene.addItem(self.all_walls_item)
        
        pen = QPen(Qt.black)
        brush = QBrush(Qt.red)
        self.door_items = set()
        self.all_doors_item = QGraphicsItemGroup()
        logger.debug("Creating %d door items", len(scene.map_doors))
        for door in scene.map_doors:
            # Doors have been expanded to individual lines for ease of fog
            # calculation, convert to polyline
            points = [QPointF(door[(i*4) % len(door)], door[(i*4+1) % len(door)]) for i in xrange(len(door)/4)]
            points.append(QPointF(door[-2], door[-1]))
            
            item = QGraphicsPolygonItem(QPolygonF(points))
            item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable)
            item.setPen(pen)
            item.setBrush(brush)
            item.setData(0, door)
            self.door_items.add(item)
            self.all_doors_item.addToGroup(item)

        gscene.addItem(self.all_doors_item)

        self.updateFog(g_draw_map_fog, g_blend_map_fog)
        
    def updateFog(self, draw_map_fog, blend_map_fog):
        map_fog_item = self.map_fog_item
        map_token_item = self.map_token_item
        gscene = self.gscene
        scene = self.scene
        token_pos = map_token_item.sceneBoundingRect().center()
        token_x, token_y = token_pos.x(), token_pos.y()
        logger.info("updateFog %f %f", token_x, token_y)

        logger.info("removing old fog")
        if (map_fog_item is not None):
            gscene.removeItem(map_fog_item)

        fog = None
        if (draw_map_fog):
            # XXX This is not correct, the depth of the frustum required to
            #     cover the whole image is actually infinity because of the
            #     degenerate case when the token is very close to the wall and
            #     the frustum angle is 180 degrees. The solution is to cap the
            #     polygon with the image bounds
            l = math.sqrt(gscene.width() ** 2 + gscene.height() ** 2)

            logger.info("draw fog %d polys pos %s", len(scene.map_walls), (token_x, token_y) )
            
            # XXX This could use degenerate triangles to merge polygons into
            #     a single call?
            # Define the brush (fill).
            if (blend_map_fog):
                brush = QBrush(QColor(0, 0, 255, 125))
                pen = QPen(Qt.black)

            else:
                #brush.setColor(QColor(0, 0, 0))
                brush = QBrush(QColor(196, 196, 196))
                pen = QPen(QColor(196, 196, 196))

            fog = QGraphicsItemGroup()
            max_dist = (60 * scene.cell_diameter / 5.0) ** 2
            for item in self.wall_items | self.door_items:
                wall_or_door = item.data(0)

                # Doors are polylines, walls are lines
                if (item not in self.open_door_items):
                    for i in xrange(0, len(wall_or_door), 4):
                    
                        x0, y0, x1, y1 = wall_or_door[i:i+4]

                        if False and ( 
                            (((x0 - token_x) ** 2 + (y0 - token_y) ** 2) > max_dist) and
                            (((x1 - token_x) ** 2 + (y1 - token_y) ** 2) > max_dist)
                            ):
                            continue
                        
                        frustum = [
                            (x0, y0), 
                            (x0 + (x0 - token_x) * l, (y0 - token_y) * l),
                            (x1 + (x1 - token_x) * l, (y1 - token_y) * l), 
                            (x1, y1)
                        ]
                        
                        item = QGraphicsPolygonItem(QPolygonF([QPointF(p[0], p[1]) for p in frustum]))
                        item.setBrush(brush)
                        item.setPen(pen)
                        fog.addToGroup(item)
                    
            gscene.addItem(fog)
            
        self.map_fog_item = fog

    def updateImage(self):
        logger.info("updateImage")
        global img_bytes
        gscene = self.gscene
        gscene.clearSelection()
        img_scale = 0.5
        img_scale = 1.5
        #scene.setSceneRect(scene.itemsBoundingRect())
        
        qim = QImage(gscene.sceneRect().size().toSize() * img_scale, QImage.Format_ARGB32)
        qim.fill(Qt.transparent)
        p = QPainter(qim)
        # XXX Ideally this should just toggle the fog item, but that also
        #     requires the fog to be always calculated even if g_draw_map_fog is
        #     disabled?
        self.updateFog(True, False)
        

        # Hide all DM user interface helpers
        # XXX Hiding seems to be slow, verify? Try changing all to transparent
        #     otherwise? Have a player and a DM scene?
        logger.info("hiding DM ui")
        if (g_draw_walls):
            self.all_walls_item.hide()
        self.all_doors_item.hide()
        logger.info("Rendering %d scene items on %dx%d image", len(gscene.items()), qim.width(), qim.height())
        gscene.render(p)
        
        # Restore all DM user interface helpers
        logger.info("restoring DM ui")
        if (g_draw_walls):
            self.all_walls_item.show()
        self.all_doors_item.show()

        self.updateFog(g_draw_map_fog, g_blend_map_fog)
        # This is necessary so the painter winds down before the pixmap
        # below, otherwise it crashes with "painter being destroyed while in
        # use"
        p.end()
        
        # convert QPixmap to PNG or JPEG bytes
        # XXX This should probably be done in the http thread and cached?
        #     But needs to check pixmap affinity or pass bytes around, also
        #     needs to check Qt grabbing the lock and using Qt from non qt 
        #     thread
        logger.info("Storing into %s buffer", imformat)
        ba = QByteArray()
        buff = QBuffer(ba)
        buff.open(QIODevice.WriteOnly) 
        ok = qim.save(buff, imformat)
        assert ok
        img_bytes = ba.data()
            
        logger.info("Converting to pixmap")
        pix = QPixmap.fromImage(qim)

        # Make sure the token is visible
        # XXX This should probably track the active from many map tokens or have
        #     more options so line of sight can be switched to other tokens?
        self.imageWidget.setPixmap(pix)
        self.imageWidget.ensureVisible(
            ((self.map_token_item.scenePos().x() - self.gscene.sceneRect().x()) * img_scale) * self.imageWidget.scale, 
            ((self.map_token_item.scenePos().y() - self.gscene.sceneRect().y()) * img_scale) * self.imageWidget.scale,
            self.imageWidget.width() / 4.0, 
            self.imageWidget.height() / 4.0
        )
        
    
    def eventFilter(self, source, event):
        logger.debug("eventFilter source %r type %d", source, event.type())
        
        if ((event.type() == QEvent.GraphicsSceneMouseMove) and 
            (self.gscene.mouseGrabberItem() is not None)):
            
            # XXX Create an inherited class and check drag events on items, snap
            #     to grid, draw a grid, etc instead of using the eventfilter

            # Note when the token is a group, the group children are grabbed,
            # not the group
            if (self.map_token_item in [self.gscene.mouseGrabberItem().group(), self.gscene.mouseGrabberItem()]):
                # Track the token on the player view
                # XXX Move this to somewhere around updateImage
                self.imageWidget.ensureVisible(
                    ((self.map_token_item.scenePos().x() - self.gscene.sceneRect().x()) / 2.0) * self.imageWidget.scale, 
                    ((self.map_token_item.scenePos().y() - self.gscene.sceneRect().y()) / 2.0) * self.imageWidget.scale,
                    self.imageWidget.width() / 4.0, 
                    self.imageWidget.height() / 4.0
                )
                # No need to update the fog since it will be done when
                # updateImage toggles the fog on and (maybe) off below

            # Update the image if any token (hero or not) moved
            self.updateImage()
            
        elif ((event.type() == QEvent.GraphicsSceneMouseDoubleClick) and 
            (self.gscene.focusItem() is not None)):
            focusItem = self.gscene.focusItem()

            if (focusItem in self.door_items):
                logger.info("Door!")
                if (focusItem in self.open_door_items):
                    self.open_door_items.remove(focusItem)
                    focusItem.setBrush(QBrush(Qt.red))
                    
                else:
                    self.open_door_items.add(focusItem)
                    focusItem.setBrush(QBrush(Qt.green))

                # No need to update fog since it will be done by updateImage
                # toggles on and (maybe) off
                self.updateImage()
                    

        # XXX Not clear this is best as a viewport() or scene() eventfilter
        #     (as a graphicsview eventfilter doesn't work because the scrollarea
        #     catches the scroll wheel event before the graphicsview)
        ## if ((event.type() == QEvent.Wheel) and (event.modifiers() == Qt.ControlModifier)):
        elif (event.type() == QEvent.GraphicsSceneWheel):
            logger.debug("Wheel 0x%x", event.modifiers())
            # Zoom Factor
            zoomFactor = 1.0015 ** event.delta() 
            
            # Set Anchors
            
            # XXX Empirically AnchorUnderMouse drifts slightly several pixels,
            #     use NoAnchor and do the calculation manually
            self.graphicsView.setTransformationAnchor(QGraphicsView.NoAnchor)
            self.graphicsView.setResizeAnchor(QGraphicsView.NoAnchor)
            
            pos = self.graphicsView.mapFromGlobal(event.screenPos())
            
            # Save scale and translate back
            oldPos = self.graphicsView.mapToScene(pos)
            self.graphicsView.scale(zoomFactor, zoomFactor)
            newPos = self.graphicsView.mapToScene(pos)
            delta = newPos - oldPos
            self.graphicsView.translate(delta.x(), delta.y())
            
            # Prevent propagation, not enough with returning True below
            event.accept()
            
            return True
    
        elif ((event.type() == QEvent.KeyPress) and (source is self.imageWidget)):
            logger.info("eventFilter %r key %d text %r", event.text(), event.key(), event.text())
            if (event.text() == "f"):
                self.imageWidget.setFitToWindow(not self.imageWidget.fitToWindow)
                self.imageWidget.update()
        
        elif ((event.type() == QEvent.KeyPress) and (source is self.gscene)):
            logger.info("eventFilter %r key %d text %r", event.text(), event.key(), event.text())

            # XXX This should snap to grid
            if (event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]):
                d = { Qt.Key_Left : (-1, 0), Qt.Key_Right : (1, 0), Qt.Key_Up : (0, -1), Qt.Key_Down : (0, 1)}
                snap_granularity = self.scene.cell_diameter / 2.0
                move_granularity = snap_granularity
                delta = QPointF(*d[event.key()]) * move_granularity

                # Snap and move the new position
                new_pos = ((self.map_token_item.pos() + delta) / snap_granularity).toPoint() * snap_granularity

                # Intersect the path against the existing walls and doors, abort
                # the movement if it crosses one of those

                # Use a bit of slack by an extra half movement unit to avoid
                # getting stuck in the intersection point due to floating point
                # precision issues
                l = QLineF(
                    self.map_token_item.sceneBoundingRect().center(), 
                    self.map_token_item.sceneBoundingRect().center() + delta * 1.10)

                # Note walls are lines, not polys so need to check line to line
                # intersection instead of poly to line. Even if that wasn't the
                # case, PolygonF.intersect is new in Qt 5.10 which is not
                # available on this installation, so do only line to line
                # intersection
                i = QLineF.NoIntersection
                for wall_item in self.wall_items:
                    i = l.intersect(wall_item.line(), None)
                    ll = wall_item.line()
                    logger.debug("wall intersection %s %s is %s", l, ll, i)
                    
                    if (i == QLineF.BoundedIntersection):
                        logger.debug("Aborting token movement, found wall intersection %s between %s and %s", i, l, ll)
                        break
                else:
                    # Check closed doors
                    # XXX intersects is not on this version of Qt (5.10), roll
                    #     manual checks
                    for door_item in self.door_items:
                        if (door_item in self.open_door_items):
                            continue
                        p0 = door_item.polygon().at(0)
                        # XXX This could early discard doors that are too far
                        #     away/don't intersect the bounding box, etc
                        for p in door_item.polygon():
                            ll = QLineF(p0, p)
                            i = l.intersect(ll, None)
                            logger.debug("door intersection %s %s is %s", l, ll, i)
                            
                            if (i == QLineF.BoundedIntersection):
                                logger.debug("Aborting token movement, found door intersection %s between %s and %s", i, l, ll)
                                break
                            p0 = p
                        if (i == QLineF.BoundedIntersection):
                            break
                    else:
                        self.map_token_item.setPos(new_pos)
                    
                self.graphicsView.ensureVisible(self.map_token_item, self.graphicsView.width()/4.0, self.graphicsView.height()/4.0)
                self.updateImage()
                return True
                
            elif (event.text() == " "):
                # Open the adjacent door
                threshold2 = (self.scene.cell_diameter ** 2.0) * 1.1
                token_center = self.map_token_item.sceneBoundingRect().center()
                for door_item in self.door_items:
                    door_center = door_item.sceneBoundingRect().center()
                    v = (door_center - token_center)
                    dist2 = QPointF.dotProduct(v, v)
                    logger.info("checking token %s vs. door %s %s vs. %s", token_center, door_center, dist2, threshold2)
                    if (dist2 < threshold2):
                        if (door_item in self.open_door_items):
                            self.open_door_items.remove(door_item)
                            door_item.setBrush(QBrush(Qt.red))
                        
                        else:
                            self.open_door_items.add(door_item)
                            door_item.setBrush(QBrush(Qt.green))
                        
                        break

            elif (event.text() == "f"):
                global g_draw_map_fog
                g_draw_map_fog = not g_draw_map_fog
            
            elif (event.text() == "w"):
                global g_draw_walls
                g_draw_walls = not g_draw_walls
                # Always draw doors otherwise can't interact with them
                # XXX Have another option? Paint them transparent?
                # XXX Is painting transparent faster than showing and hiding?
                if (g_draw_walls):
                    self.all_walls_item.show()

                else:
                    self.all_walls_item.hide()

            elif (event.text() == "m"):
                if (self.player.state() != QMediaPlayer.PlayingState):
                    self.player.play()
                    logger.info("Playing %r", self.playlist.currentMedia().canonicalUrl())
                    
                else:
                    self.player.pause()
                    self.playlist.next()


            elif (event.text() == "b"):
                global g_blend_map_fog
                g_blend_map_fog = not g_blend_map_fog

            # No need to update the fog since it will be done by updateImage
            # when it toggles the fog on and (maybe) off
            self.updateImage()
            
        return super(VTTMainWindow, self).eventFilter(source, event)

            
def main():
    report_versions()
    thread.start_new_thread(server_thread, (None,))

    app = QApplication(sys.argv)
    ex = VTTMainWindow()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()