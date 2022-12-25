#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

"""
Qt Virtual Table Top
(c) Antonio Tejada 2022

"""

import csv
import io
import json
import logging
import math
import os
import random
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
    """
    Split lines in multiple records, fill in %(className)s
    """
    def __init__(self):
        super(LineHandler, self).__init__()

    def emit(self, record):
        # Find out class name, _getframe is supposed to be faster than inspect,
        # but less portable
        # caller_locals = inspect.stack()[6][0].f_locals
        caller_locals = sys._getframe(6).f_locals
        clsname = ""
        zelf = caller_locals.get("self", None)
        if (zelf is not None):
            clsname = class_name(zelf) + "."
            zelf = None
        caller_locals = None
        
        # Indent all lines but the first one
        indent = ""
        text = record.getMessage()
        messages = text.split('\n')
        for message in messages:
            r = record
            r.msg = "%s%s" % (indent, message)
            r.className = clsname
            r.args = None
            super(LineHandler, self).emit(r)
            indent = "    " 

def setup_logger(logger):
    """
    Setup the logger with a line break handler
    """
    logging_format = "%(asctime).23s %(levelname)s:%(filename)s(%(lineno)d):[%(thread)d] %(className)s%(funcName)s: %(message)s"

    logger_handler = LineHandler()
    logger_handler.setFormatter(logging.Formatter(logging_format))
    logger.addHandler(logger_handler) 

    return logger

logger = logging.getLogger(__name__)
setup_logger(logger)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)

class Struct:
    """
    A structure that can have any fields defined or added.

        s = Struct(field0=value0, field1=Value1)
        s.field2 = value2
    
    or

        s = Struct(**{field0 : value0, field1=value1})
    
    """
    def __init__(self, **entries): 
        self.__dict__.update(entries)


class JSONStructEncoder(json.JSONEncoder):
    """
    JSON Encoder for Structs
    """
    def default(self, obj):
        if (isinstance(obj, Struct)):
            return obj.__dict__

        else:
            return json.JSONEncoder.default(self, obj)


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

def print_gc_stats(collect = True):
    import gc
    logger.info("GC stats")

    logger.info("garbage %r", gc.garbage)
    logger.info("gc counts %s", gc.get_count())
    if (collect):
        logger.info("gc collecting")
        gc.collect()
        logger.info("gc collected")
        logger.info("garbage %r", gc.garbage)
        logger.info("gc counts %s", gc.get_count())

def os_copy(src_filepath, dst_filepath):
    logger.info("Copying from %r to %r", src_filepath, dst_filepath)
    with open(src_filepath, "rb") as f_src, open(dst_filepath, "wb") as f_dst:
        chunk_size = 4 * 2 ** 10
        chunk = None
        while (chunk != ""):
            chunk = f_src.read(chunk_size)
            f_dst.write(chunk)

def os_path_normall(path):
    return os.path.normcase(os.path.normpath(path))

def os_path_isroot(path):
    # Checking path == os.path.dirname(path) is the recommended way
    # of checking for the root directory
    # This works for both regular paths and SMB network shares
    dirname = os.path.dirname(path)
    return (path == dirname)


def os_path_abspath(path):
    # The os.path library (split and the derivatives dirname, basename) slash
    # terminates root directories, eg
    
    #   os.path.split("\\") ('\\', '')
    #   os.path.split("\\dir1") ('\\', 'dir1')
    #   os.path.split("\\dir1\\dir2") ('\\dir1', 'dir2')
    #   os.path.split("\\dir1\\dir2\\") ('\\dir1\\dir2', '')
    
    # this includes SMB network shares, where the root is considered to be the
    # pair \\host\share\ eg 
    
    #   os.path.split("\\\\host\\share") ('\\\\host\\share', '')
    #   os.path.split("\\\\host\\share\\") ('\\\\host\\share\\', '')
    #   os.path.split("\\\\host\\share\\dir1") ('\\\\host\\share\\', 'dir1')

    # abspath also slash terminates regular root directories, 
    
    #  os.path.abspath("\\") 'C:\\'
    #  os.path.abspath("\\..") 'C:\\'

    # unfortunately fails to slash terminate SMB network shares root
    # directories, eg
    
    #  os.path.abspath("\\\\host\\share\\..") \\\\host\\share
    #  os.path.abspath("\\\\host\\share\\..\\..") '\\\\host\\share

    # Without the trailing slash, functions like isabs fail, eg

    #   os.path.isabs("\\\\host\\share") False
    #   os.path.isabs("\\\\host\\share\\") True
    #   os.path.isabs("\\\\host\\share\\dir") True
    #   os.path.isabs("\\\\host\\share\\..") True
    
    # See https://stackoverflow.com/questions/34599208/python-isabs-does-not-recognize-windows-unc-path-as-absolute-path

    
    # This fixes that by making sure root directories are always slash
    # terminated
    abspath = os.path.abspath(os.path.expanduser(path))
    if ((not abspath.endswith(os.sep)) and os_path_isroot(abspath)):
        abspath += os.sep

    logger.info("%r is %r", path, abspath)
    return abspath


def index_of(l, item):
    """
    Find the first occurrence of item in the list or return -1 if not found
    """
    try:
        return l.index(item)

    except ValueError:
        return -1

def find_parent(ancestor, item):
    logger.info("Looking for %s in %s", item, ancestor)
    if (isinstance(ancestor, (list, tuple))):
        for value in ancestor:
            if (value is item):
                return ancestor
            l = find_parent(value, item)
            if (l is not None):
                return l

    elif (hasattr(ancestor, "__dict__")):
        for value in ancestor.__dict__.values():
            if (value is item):
                return ancestor
            l = find_parent(value, item)
            if (l is not None):
                return l
    
    return None

def class_name(o):
    return o.__class__.__name__

def qSizeToPointF(size):
    return QPointF(size.width(), size.height())

def qSizeToPoint(size):
    return QPoint(size.width(), size.height())

def qtuple(q):
    """
    Convert Qt vectors (QPoint, etc) to tuples
    """
    if (isinstance(q, (QPoint, QPointF))):
        return (q.x(), q.y())

    elif (isinstance(q, (QSize, QSizeF))):
        return (q.width(), q.height())

    elif (isinstance(q, (QLine, QLineF))):
        return (qtuple(q.p1()), qtuple(q.p2()))

    elif (isinstance(q, (QRect, QRectF))):
        # Note this returns x,y,w,h to match Qt parameters
        return (qtuple(q.topLeft()), qtuple(q.size()))

    else:
        assert False, "Unhandled Qt type!!!"

eventTypeToName = { getattr(QEvent, name) : name for name in vars(QEvent) if type(getattr(QEvent, name)) == QEvent.Type}
def qEventTypeToString(eventType):

    return eventTypeToName.get(eventType, str(eventType))

def qFindTabBarFromDockWidget(dock):
    logger.info("%s", dock.windowTitle())
    # QDockWidget tabs are QTabBars children of the main window, in C++ the dock
    # can be found by checking that the QDockWidget address matches
    # tab.tabData(), unfortunately there doesn't seem to be a way to get the C++
    # QDockWidget pointer from the Python QDockWidget wrapper. Instead, this
    # checks the tabData bounds vs. the dock bounds plus some slack to find the
    # tab that geometrically matches the dock.
    #
    # See https://bugreports.qt.io/browse/QTBUG-40913 See
    # https://www.qtcentre.org/threads/61471-(pyqt)-Identify-QDockWidget-by-QTabBar-created-automatically-by-QMainWindow-in-PyQT
    for tabBar in dock.parent().findChildren(QTabBar):
        if ((tabBar.geometry().x() == dock.geometry().x()) and 
            (tabBar.geometry().width() == dock.geometry().width()) and
            # This is actually 2, use a bit of slack
            ((tabBar.geometry().y() - dock.geometry().bottom()) < 10)
            ):
            return tabBar

    return None

class NumericTableWidgetItem(QTableWidgetItem):
    """
    See https://stackoverflow.com/questions/25533140/sorting-qtablewidget-items-numerically
    """
    def __init__ (self, value=""):
        super(NumericTableWidgetItem, self).__init__('%s' % value)

    def __lt__ (self, other):
        if (isinstance(other, NumericTableWidgetItem)):
            selfDataValue  = self.text()
            selfDataValue = 0 if (selfDataValue == "") else float(selfDataValue)
            otherDataValue = other.text()
            otherDataValue = 0 if (otherDataValue == "") else float(otherDataValue)
            
            return selfDataValue < otherDataValue
        else:
            return super(NumericTableWidgetItem, self).__lt__(self, other)

class LinkTableWidgetItem(QTableWidgetItem):
    def __init__(self, value = "", link = None):
        super(LinkTableWidgetItem, self).__init__("%s" % value)
        self.setLink(link)

    def link(self):
        return self.link

    def setLink(self, link = None):
        self.link = link
        if (link is not None):
            font = self.font()
            font.setUnderline(True)
            self.setFont(font)
            # XXX This should get it from some global palette?
            self.setForeground(QBrush(Qt.blue))
        # XXX Restore font & foreground if link is None?

def import_ds(scene, filepath):
    # Load a dungeonscrawl .ds data file 
    # See https://www.dungeonscrawl.com/
    
    # Recent ds files are zip compressed with a "map" file inside, try to
    # uncompress first, then parse
    try:
        f = zipfile.ZipFile(filepath)
        # The zip has one member with the name "map"
        f = f.open("map")

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
        selected_page = js["state"]["document"]["nodes"]["document"]["selectedPage"]
        map_cell_diameter = float(js["state"]["document"]["nodes"][selected_page]["grid"]["cellDiameter"])
        
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
                    map_doors.append(Struct(lines=door, open=False))
                
        # These correspond to the door handles, adding as walls
        # XXX Do a better parse that doesn't assume two lines per entry?
        map_walls.extend([line[0] + line[1] for line in lines])

    logger.info("Found %d doors %d valid walls, %d coalesced, %d duplicated", len(map_doors), len(map_walls), 
        coalesced_walls, duplicated_walls)
    
    scene.cell_diameter = map_cell_diameter
    scene.map_walls = map_walls
    scene.map_doors = map_doors
    scene.music = []

    return scene


def load_ds(ds_filepath, map_filepath=None, img_offset_in_cells=None):
    scene = Struct()

    import_ds(scene, ds_filepath)

    if (map_filepath is not None):

        # Size in cells is embedded in the filename 
        m = re.match(r".*\D+(\d+)x(\d+)\D+", map_filepath)
        img_size_in_cells = (int(m.group(1)), int(m.group(2)))
        logger.debug("img size in cells %s", img_size_in_cells)

        if (img_offset_in_cells is None):
            # Make a best guess at the alignment matching the center of the wall 
            # bounds with the center of the grids
            bounds = (
                min([min(wall[0], wall[2]) for wall in scene.map_walls]),
                min([min(wall[1], wall[3]) for wall in scene.map_walls]),
                max([max(wall[0], wall[2]) for wall in scene.map_walls]),
                max([max(wall[1], wall[3]) for wall in scene.map_walls])
            )
            bounds = QRectF(QPointF(*bounds[0:2]), QPointF(*bounds[2:4]))
            img_size = QSizeF(*img_size_in_cells) * scene.cell_diameter
            margin = (img_size - bounds.size()) / 2.0
            img_offset_in_cells = [round(c) for c in qtuple(-(bounds.topLeft() - qSizeToPointF(margin)) / scene.cell_diameter)]

        scene.map_images = [
            Struct(**{
                # XXX Some dungeon scrawl v1 png files are saved with the wrong pixel
                #     height, ideally should do rounding at png loading time?
                "scale" : scene.cell_diameter * img_size_in_cells[0],
                "filepath" : os.path.relpath(map_filepath),
                "scene_pos" : qtuple(-QPointF(*img_offset_in_cells) * scene.cell_diameter)
                # XXX This assumes homogeneous scaling, may need to use scalex,y
            })
        ]

    # Create some dummy tokens for the time being
    # XXX Remove once tokens can be imported or dragged from the token browser
    scene.map_tokens = []
    for filename in [
        "Hobgoblin.png", "Hobgoblin.png", "Goblin.png" ,"Goblin.png", 
        "Goblin.png", "Gnoll.png", "Ogre.png", "Ancient Red Dragon.png", 
        "Knight.png", "Priest.png", "Mage.png"]:
        map_token = Struct()
        map_token.filepath = os.path.join("_out", "tokens", filename)
        map_token.scene_pos = (0.0, 0.0)
        map_token.name = os.path.splitext(filename)[0]
        map_token.hidden = False
        map_token.ruleset_info = Struct(**default_ruleset_info)
        
        pix_scale = scene.cell_diameter
        # XXX This should check the monster size or such
        if ("Dragon" in filename):
            pix_scale *= 4.0
        elif ("Ogre" in filename):
            pix_scale *= 1.5

        map_token.scale = pix_scale

        scene.map_tokens.append(map_token)    

    return scene


def build_index(dirpath):
    logger.info("%r", dirpath)
    words = dict()
    filepath_to_title = dict()
    title_to_filepath = dict()
    
    for subdirpath in ["Monsters1", R"cdrom\WEBHELP\DMG", R"cdrom\WEBHELP\PHB"]:
        print "indexing", subdirpath
        for filename in os.listdir(os.path.join(dirpath, subdirpath)):
            if (filename.lower().endswith((".htm", ".html"))):
                with open(os.path.join(dirpath, subdirpath, filename), "r") as f:
                    print "reading", filename

                    # Use lowercase for paths and words so searches can be
                    # case-insenstive and so paths work after normalization or
                    # retrieval from browser which would lowercase them
                    subfilepath = os.path.join(subdirpath, filename).lower()

                    s = f.read()
                    m = re.search("<TITLE>([^<]+)</TITLE>", s, re.IGNORECASE)
                    if (m is not None):
                        title = m.group(1)
                        filepath_to_title[subfilepath] = title
                        title_to_filepath[title] = subfilepath
                    
                    # Remove HTML tags so they are not indexed
                    # XXX Note this needs a Qapp created or it will exit
                    #     without any warnings
                    frag = QTextDocumentFragment.fromHtml(s)
                    s = frag.toPlainText()
                    
                    print "tokenizing", filename
                    for word in re.split(r"\W+", s):
                        if (word != ""):
                            word = word.lower()
                            ss = words.get(word, set())
                            ss.add(subfilepath)
                            words[word] = ss
                    print "tokenized"
        
    # XXX Also look at the index file cdrom\WEBHELP\INDEX.HHK which is XML
    # XXX Could gzip it
    with open(os.path.join("_out", "index.json"), "w") as f:
        json.dump(
            { 
                "word_to_filepaths" : {key : list(words[key]) for key in words }, 
                "filepath_to_title" : filepath_to_title,
                "title_to_filepath" : title_to_filepath
            }, f, indent=2
        )


def eval_dice(s):
    """
    Evaluate a die expression eg 1d20, 2d4+1, 1d4 * 3 + 8, (1d4 + 1) * 1d5
    - Valid operations are - + * / ()
    - Valid operands are number or dice, where dice is "NdM" with n the number
      of dice and M the sides in the die 
    """
    def get_token(state):
        """
        Return a token and its integer value if it's a die or an integer, or 
        None if no more tokens are available

        Valid tokens are + - * / ( ) number die_expression, white space is
        ignored

        """
        # Consume whitespace
        while ((state.string_index < len(state.string)) and 
               state.string[state.string_index].isspace()):
            state.string_index += 1

        state.token = Struct()
        if (state.string_index >= len(state.string)):
            state.token.type = None
            
        else:
            if (state.string[state.string_index] in ["+", "-", "*", "/", "(", ")"]):
                state.token.type = state.string[state.string_index]
                state.string_index += 1

            else:
                # NdM expression eg 1d10, 5d6, etc
                m = re.match(r"(\d+)d(\d+)", state.string[state.string_index:])
                if (m is not None):
                    state.token.type = "die"
                    state.token.value = int(m.group(1)) * random.randint(1, int(m.group(2)))
                    
                else:
                    # Number
                    m = re.match(r"(\d+)", state.string[state.string_index:])
                    state.token.type = "number"
                    state.token.value = int(m.group(1))

                state.string_index += len(m.group(0))

            logger.info("read token %s", state.token.type)
        
        return state.token
    
    def eval_expr(state):
        """
        Evaluate general expression 

        expr :
              mult_expr op_add expr
              mult_expr
        """
        value = eval_mult_expr(state)
        if (state.token.type == "+"):
            get_token(state)
            value += eval_expr(state)

        elif (state.token.type == "-"):
            get_token(state)
            value -= eval_expr(state)

        return value

    def eval_mult_expr(state):
        """
        Evaluate multiplicative expression
        
        mult_expr : 
             simple_expr op_mul mult_expr
             simple_expr
        """
        value = eval_simple_expr(state)
        if (state.token.type == "*"):
            get_token(state)
            value *= eval_mult_expr(state)

        elif (state.token.type == "/"):
            get_token(state)
            value /= eval_mult_expr(state)

        return value

    def eval_simple_expr(state):
        """
        Evaluate simple expression

        simple_expr : 
              die_expr         eg 1d6
              number           eg 1
              ( expr )
        
        XXX Could also allow variables here this.AC, etc
        """
        token = state.token
        if (token.type in ["die", "number"]):
            get_token(state)
            value = token.value
            
        elif (token.type == "("):
            get_token(state)
            value = eval_expr(state)
            assert state.token.type == ")"
            get_token(state)

        else:
            assert False, "Wrong dice expression"

        return value
        
    logger.info("%s", s)

    state = Struct()
    state.string = s
    state.string_index = 0
    token = get_token(state)

    value = 0
    if (token.type is not None):
        value = eval_expr(state)
        # Expect to consume all tokens and the whole string
        assert state.token.type is None

    logger.info("value %d", value)
    return value


def qPopulateTable(table, rows):
    """
    Populate a Qt table from a list of rows, with the first row being the
    headers
    """
    # Disable sorting otherwise items will sort themselves mid-insertion
    # interfering with insertion
    table.setSortingEnabled(False)

    # Remove all rows from the table
    table.setRowCount(0)
    for j, row in enumerate(rows):
        if (j == 0):
            # Headers
            table.setRowCount(len(rows)-1)
            table.setColumnCount(len(row))
        
        for i, cell in enumerate(row):
            item = QTableWidgetItem(cell)
            if (j == 0):
                table.setHorizontalHeaderItem(i, item)

            else:
                table.setItem(j-1, i, item)

    table.setSortingEnabled(True)


class VTTTableWidget(QTableWidget):
    """
    Options for supporting links on cells:

    Needs to support:
      - mouse cursor change to hand
      - edit on double click, other cell navigation (tab navigates/edits next
        cell, up/down navigates up/down, esc aborts edit)
      - navigation on single click
      - navigation on tab+enter
      - sorting

    setCellWidget QLabel with HTML contents:
      - easy, just works for readonly
      - switches to hand icon
      - keyboard navigation
      - not editable, but if you set Qt.TextEditorInteraction then you can edit,
        would need tweaking to edit only on double click, etc
      - sorting doesn't work (probably sorts by html link, could be fixed with Qt.InitialSortRole?)
      - focused text color doesn't change to white

    setCellWidget QLabel + itemdelegate
      - No need to calculate text extents for hand cursor
      - No need
      - Very messy, needs two unrelated objects

    setBrush+font underline + viewport mouse hover eventFilter + table
    eventfilter
      - created items need to set the color and underline on the item
      - viewport eventfilter for mouse hover and click
      - table eventfilter for keyboard
      - simple and sorting just works
      - not very encapsulated unless the table is subclassed?

    setCellWidget QLabel + lineedit
      - Needs to mimic all keys from itemdelegate (tab, up arrow, etc)

    column Itemdelegate + editorEvent
      - editorEvent captures the click and the mousemovement/cursor change ok
      - In addition to the delegate, needs to set the underline and font color
        on ItemDelegate::paint or at item creation
      - not very clean, the itemdelegate is global but needs to keep checking at
        paint time whether the item 
      - editorevent doesn't capture keystrokes when not in editing mode, can't
        open urls with keyboard unless eventFilter/etc is set too

    QTableWidgetItem subclass
      - Doesn't seem like it does much, there's no hook for when the item is
        added to the table

    QTExtEdit with HTML contents
      - can't easily trap link clicking?
      - doesn't switch to hand icon?

    XXX Use an item delegate and editorEvent instead of this complicated
    delegate + label See
    https://forum.qt.io/topic/23808/solved-qstyleditemdelegate-custom-mouse-cursor-for-tablewidget-item/3

    """
    linkActivated = pyqtSignal(str)
    def __init__(self, parent=None):
        logger.info("")
        super(VTTTableWidget, self).__init__(parent)
        
        # Cells with links require trapping mouse hovering and clicking on them,
        # and pressing return/ctrl+return on cells
        self.viewport().installEventFilter(self)
        self.setMouseTracking(True)
        self.installEventFilter(self)

    def itemLink(self, item):
        logger.info("%s", None if item is None else item.text())
        link = None if item is None else getattr(item, "link", None)
        return link

    def eventFilter(self, source, event):
        # This can also be done with the individual event handlers, but using
        # eventFilter allows a unique codepath and probably simpler (the
        # individual event handler won't trap eg viewport events)
        logger.info("source %s type %s", class_name(source), qEventTypeToString(event.type()))
        table = self
        if (
            ((source == table.viewport()) and 
             (event.type() in [QEvent.Leave, QEvent.MouseMove, QEvent.MouseButtonPress])) or
             # Don't trap keys when the table doesn't have the focus (eg when
             # the cell editor is open)
            ((source == table) and (event.type() == QEvent.KeyPress) and 
             (event.key() == Qt.Key_Return) and (table.hasFocus()))
        ):
            if (event.type() == QEvent.Leave):
                table.unsetCursor()
            
            elif (event.type() == QEvent.KeyPress):
                item = table.currentItem()
                
                link = self.itemLink(item)
                if (link is not None):
                    self.linkActivated.emit(link)
                    return True 

            elif (event.type() in [QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.KeyPress]):
                item = table.itemAt(event.pos())
                link = self.itemLink(item)
                if (link is not None):
                    cellRect = QRectF(
                        table.columnViewportPosition(item.column()), 
                        table.rowViewportPosition(item.row()), 
                        table.columnWidth(item.column()), 
                        table.rowHeight(item.row())
                    )

                    # The item's font size is 7.8, the table's is 9.0 but
                    # visually the item is being rendered with 9.0, so use the
                    # table's for the font metrics rather than the item's

                    # XXX Another way of fixing this is to re-set the table's
                    #     font point size to the same table's font point size
                    #     value (!), which will make the item then render at 7.8
                    #     (!)
                    fm = QFontMetricsF(item.font())
                    fm = QFontMetricsF(table.font())
                    logger.info("item size %f table size %f", item.font().pointSizeF(), table.font().pointSizeF())

                    # XXX Get alignments from somewhere
                    flags= Qt.AlignLeft | Qt.AlignVCenter

                    rect = fm.boundingRect(cellRect, flags, item.text())
                    logger.info("Item %s pos %s rect %s", item.text(), qtuple(event.pos()), qtuple(rect))
                    if (rect.contains(event.pos())):
                        table.setCursor(Qt.PointingHandCursor)
                        logger.info("inside")
                        if (event.type() == QEvent.MouseButtonPress):
                            token = item.data(Qt.UserRole)
                            # Ignore this click, especially if ctrl is pressed
                            # for "open in new browser", don't want to select
                            # the clicked cell
                            self.linkActivated.emit(self.itemLink(item))
                            return True

                    else:
                        table.unsetCursor()

        return super(VTTTableWidget, self).eventFilter(source, event)
    
class CombatTracker(QWidget):
    sceneChanged = pyqtSignal()
    browseMonster = pyqtSignal(str)

    def __init__(self, parent=None):
        super(CombatTracker, self).__init__(parent)

        self.updatesBlocked = False

        headers = [
            "Name", "PC", "I", "I+", "A(D)", "A+", "HD", "HP", "HP_", "AC", 
            "MR", "T0", "#AT", "Damage", "Alignment", "Notes"
        ]
        self.numericHeaders = set(["I", "HD", "HP", "HP_", "AC", "T0", "#AT"])
        self.headerToColumn = dict()
        for i, header in enumerate(headers):
            self.headerToColumn[header] = i
        
        table = VTTTableWidget()
        self.table = table
        table.setColumnCount(len(headers))
        for i, header in enumerate(headers):
            item = QTableWidgetItem(header)
            table.setHorizontalHeaderItem(i, item)
        table.setSortingEnabled(True)
        table.cellChanged.connect(self.cellChanged)
        table.linkActivated.connect(self.browseMonster.emit)
        # Don't focus just on mouse wheel
        table.setFocusPolicy(Qt.StrongFocus)
    
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(table)
        
        hbox = QHBoxLayout()
        hbox.addStretch()
        button = QPushButton("Roll Hit Points")
        self.rollAttackButton = button
        button.clicked.connect(self.rollHitPoints)
        hbox.addWidget(button)
        hbox.setStretchFactor(button, 0)
        button = QPushButton("Roll Initiative")
        self.rollInitiativeButton = button
        button.clicked.connect(self.rollInitiative)
        hbox.addWidget(button)
        hbox.setStretchFactor(button, 0)
        button = QPushButton("Roll Attack")
        self.rollAttackButton = button
        button.clicked.connect(self.rollAttack)
        hbox.addWidget(button)
        hbox.setStretchFactor(button, 0)

        # XXX Display round number, increment round count, clear round count

        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def setUpdatesBlocked(self, blocked):
        """
        Emit a sceneChanged if updates are not blocked, this is done inside a
        blockSignals/unblockSignals bracket to prevent infinite updates when
        cellChanged triggers a sceneChanged which triggers a setSecene and
        setScene calls updateCombatTrackers which programmatically changes the
        cell and causes a cellChanged again.

        This is used in two cases:
        - prevent per cell update when modifying cells in a loop, in this case
          it's called with setUpdatesBlocked(True) at the beginning and
          setUpdatesBlocked(False) at the end which emits the sceneChanged to
          perform a scene update once all changes in the loop have been done
        - emit a sceneChanged bracketed by block/unblockSignals. In this case
          it's called with False when update is already False, in order to keep
          a single sceneChanged emit codepath.

          XXX Probably the second case could be removed once there's piecemeal
              scene updates?
        """
        self.updatesBlocked = blocked
        if (not self.updatesBlocked):
            self.table.blockSignals(True)
            self.sceneChanged.emit()
            self.table.blockSignals(False)

    def rollAttack(self):
        logger.info("")
        table = self.table

        # XXX Each cell modification causes a scene update, block updates and
        #     unblock at the end. Remove once scene updates are piecewise
        self.setUpdatesBlocked(True)

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        for j in xrange(table.rowCount()):
            # Only roll for rows without PC checked
            if (table.item(j, self.headerToColumn["PC"]).text() == ""):
                attackAdj = table.item(j, self.headerToColumn["A+"]).text()
                attackAdj = eval_dice(attackAdj)
                # - multiple damages are separated by / 
                # - individual damage is in the format die expression (eg 2d4 + 1) 
                #   or min-max (eg 3-9)
                damage = table.item(j, self.headerToColumn["Damage"]).text()
                damages = damage.split("/")
                rolls = []
                for damage in damages:
                    # Roll attack for this damage
                    attack = eval_dice("1d20") + attackAdj
                    roll = -1
                    # Check minmax format
                    m = re.search(r"(\d+)-(\d+)", damage)
                    if (m is not None):
                        logger.info("Rolling rand %s to %s", m.group(1), m.group(2))
                        roll = random.randint(int(m.group(1)), int(m.group(2)))

                    else:
                        # Check die expression
                        try:
                            roll = eval_dice(damage)
                            logger.info("Rolling die %s", damage)

                        except:
                            pass

                    rolls.append("%d (%d)" % (attack, roll))

                cell = str.join(",", rolls)
                table.item(j, self.headerToColumn["A(D)"]).setText(cell)

        table.setSortingEnabled(True)

        self.setUpdatesBlocked(False)

    def rollInitiative(self):
        logger.info("")
        table = self.table

        # XXX Each cell modification causes a scene update, block updates and
        #     unblock at the end. Remove once scene updates are piecewise
        self.setUpdatesBlocked(True)

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        for j in xrange(table.rowCount()):
            if (table.item(j, self.headerToColumn["PC"]).text() == ""):
                initiative = eval_dice("1d10")
                iAdj = table.item(j, self.headerToColumn["I+"]).text()
                if (iAdj != ""):
                    initiative += int(iAdj)

                logger.info("setting initiative for %d to %d", j, initiative)
                cell = "%d" % initiative
                
                table.item(j, self.headerToColumn["I"]).setText(cell)
        
        # XXX Display the initiative roll or the order on the tokens
        # XXX See https://game-icons.net/ for status icons

        self.setUpdatesBlocked(False)

        table.setSortingEnabled(True)
    
    def rollHitPoints(self):
        logger.info("")
        table = self.table

        # XXX Each cell modification causes a scene update, block updates and
        #     unblock at the end. Remove once scene updates are piecewise
        self.setUpdatesBlocked(True)

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        for row in xrange(table.rowCount()):
            if (table.item(row, self.headerToColumn["PC"]).text() == ""):
                token = table.item(row, 0).data(Qt.UserRole)
                ruleset_info = token.ruleset_info
                hdd = int(ruleset_info.HDD)
                hdb = int(ruleset_info.HDB)
                hd = int(ruleset_info.HD)
                hp = sum([random.randint(1, hdd) for _ in xrange(hd)]) + hdb

                logger.info("Setting hitpoints for %d to %d", row, hp)

                cell = "%d" % hp
                table.item(row, self.headerToColumn["HP"]).setText(cell)
                
                # setItem should call cellChanged which should update everything

        table.setSortingEnabled(True)

        self.setUpdatesBlocked(False)

    def cellChanged(self, row, col):
        logger.debug("row %d col %d item %s", row, col, self.table.item(row, col).text())
        table = self.table
        token = table.item(row, col).data(Qt.UserRole)
        # Token can be none when rows are being inserted, ignore update
        if (token is None):
            logger.debug("Ignoring none update")
            return
        text = table.item(row, col).text()
        header = table.horizontalHeaderItem(col).text()
        
        if (header == "Name"):
            token.name = text
            
        else:
            if (getattr(token, "ruleset_info", None) is None):
                token.ruleset_info = Struct(**default_ruleset_info)
            
            ruleset_info = token.ruleset_info
            if (header == "HD"):
                ruleset_info.HD = text

            elif (header == "#AT"):
                ruleset_info.AT = text

            elif (header == "A+"):
                ruleset_info.A_ = text

            elif (header == "HP"):
                ruleset_info.HP = text

            elif (header == "HP_"):
                ruleset_info.HP_ = text

            elif (header == "AC"):
                ruleset_info.AC = text

            elif (header == "MR"):
                ruleset_info.MR = text

            elif (header == "T0"):
                ruleset_info.T0 = text

            elif (header == "Damage"):
                ruleset_info.Damage = text

            elif (header == "Alignment"):
                ruleset_info.Alignment = text

            elif (header == "Notes"):
                ruleset_info.Notes = text

        if (not self.updatesBlocked):
            # Re set updates so the update is performed in a single codepath
            self.setUpdatesBlocked(False)

    def setScene(self, scene):
        logger.debug("scene %s", scene)
        table = self.table

        # XXX The combat tracker should contain only selected tokens unless in
        #     some "all scene tokens" or "only view tokens" mode, which would
        #     allow having multiple combat trackers eg per room in the same
        #     scene

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        # Delete removed rows, starting from the end of the table
        for i in xrange(table.rowCount()-1, -1, -1):
            # This can be None if recycling a table on a new scene?
            if ((table.item(i, 0) is None) or (table.item(i, 0).data(Qt.UserRole) not in scene.map_tokens)):
                logger.info("removing row %d", i)
                table.removeRow(i)
        
        # Modify existing and add new rows
        for token in scene.map_tokens:
            if (getattr(token, "ruleset_info", None) is not None) :
                d = {
                    "Name" : token.name,
                    "HD" : token.ruleset_info.HD,
                    "HP" : token.ruleset_info.HP,
                    "AC" : token.ruleset_info.AC,
                    "#AT" : token.ruleset_info.AT,
                    "MR" : token.ruleset_info.MR,
                    "T0" : token.ruleset_info.T0,
                    "Damage" : token.ruleset_info.Damage,
                    "Notes" : token.ruleset_info.Notes,
                    "Alignment" : token.ruleset_info.Alignment,
                }
            else:
                d = { "Name" : token.name }

            # Find the row with this token
            for i in xrange(table.rowCount()):
                if (table.item(i, 0).data(Qt.UserRole) == token):
                    logger.info("modifying row %d", i)
                    row = i
                    break
            else:
                # No row, create one
                logger.debug("creating row %d", table.rowCount())
                row = table.rowCount()
                table.insertRow(row)
                
            for i in xrange(table.columnCount()):
                header = table.horizontalHeaderItem(i).text()

                if (table.item(row, i) is None):
                    logger.debug("setting item %d, %d", row, i)
                    if (header in self.numericHeaders):
                        item = NumericTableWidgetItem()

                    elif (header == "Name"):
                        item = LinkTableWidgetItem()

                    else:
                        item = QTableWidgetItem()
                    # Note this triggers a cellChanged event, but there's no
                    # token set yet as data, so it will be ignored
                    table.setItem(row, i, item)
                
                else:
                    item = table.item(row, i)

                if (header in d):
                    cell = d[header]
                    if (i == self.headerToColumn["Name"]):
                        item.setLink(token.ruleset_info.Id)

                else:
                    cell = item.text()
                logger.debug("setting text %d, %d, %r", row, i, cell)
                # Note this triggers a cellChanged event, but there's no
                # token set yet as data, so it will be ignored
                item.setText(cell)
                logger.debug("setting data %d, %d, %s", row, i, token)
                item.setData(Qt.UserRole, token)
                

        table.setSortingEnabled(True)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()


class EncounterBuilder(QWidget):
    browseMonster = pyqtSignal(str)
    def __init__(self, parent=None):
        super(EncounterBuilder, self).__init__(parent)

        label = QLabel("Filter")
        lineEdit = QLineEdit()
        self.query = lineEdit
        self.query.setPlaceholderText("Search...")
        self.query.textChanged.connect(self.filterItems)

        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(lineEdit)
        hbox.setStretchFactor(label, 0)

        table = VTTTableWidget()
        self.monsterTable = table
        self.encounterTable = None

        filepath = os.path.join("_out", "SBLaxman's AD&D Monster List 2.1.csv")
        filepath = os.path.join("_out", "monsters2.csv")
        
        with open(filepath, "rb") as f:
            rows = list(csv.reader(f, delimiter="\t"))
            headers = rows[0]
            qPopulateTable(table, rows)

        self.monsterHeaderToColumn = dict()
        for i, header in enumerate(headers):
            self.monsterHeaderToColumn[header] = i
        # Set links to monster browser
        for row in xrange(table.rowCount()):
            item = table.item(row, self.monsterHeaderToColumn["Name"])
            table.setItem(item.row(), item.column(), LinkTableWidgetItem(
                item.text(), 
                table.item(row, self.monsterHeaderToColumn["Link"]).text()))
        table.setColumnHidden(self.monsterHeaderToColumn["Link"], True)
        # XXX This doesn't allow keyboard browseMonster because the table is
        #     readonly and return is used for transferring the monster to the
        #     encounter
        table.linkActivated.connect(self.browseMonster.emit)
        table.installEventFilter(self)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setTabKeyNavigation(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSortingEnabled(True)
        # Set the name column to stretch if the wider is larger than the table
        # Note this prevents resizing the name column, but other columns can be
        # resized and the name column will pick up the slack
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.cellDoubleClicked.connect(self.addMonsterToEncounter)
        # Don't focus just on mouse wheel
        table.setFocusPolicy(Qt.StrongFocus)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(hbox)
        vbox.addWidget(table)
        
        vsplitter = QSplitter(Qt.Vertical)
        widget = QWidget()
        widget.setLayout(vbox)
        vsplitter.addWidget(widget)

        encounterHeaders = [
            "Id", "Link", "Name", "XP", "HD", "HDB", "HDD", "HP", "HP_", "AC", 
            "MR", "T0", "#AT", "Damage", "Alignment", "Notes"
        ]
        hiddenEncounterHeaders = set(["HDB", "HDD", "Id", "Link"])
        self.encounterHeaderToColumn = dict()
        for i, header in enumerate(encounterHeaders):
            self.encounterHeaderToColumn[header] = i

        table = VTTTableWidget()
        self.encounterTable = table
        table.installEventFilter(self)
        table.setRowCount(0)
        table.setColumnCount(len(encounterHeaders))
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setSortingEnabled(True)
        table.linkActivated.connect(self.browseMonster.emit)
        # Don't focus just on mouse wheel
        table.setFocusPolicy(Qt.StrongFocus)
        
        for i, header in enumerate(encounterHeaders):
            item = QTableWidgetItem(header)
            table.setHorizontalHeaderItem(i, item)
            if (header in hiddenEncounterHeaders):
                table.hideColumn(i)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        # Don't focus just on mouse wheel
        table.setFocusPolicy(Qt.StrongFocus)

        hbox = QHBoxLayout()
        label = QLabel("Group Levels")
        hbox.addWidget(label)
        spin = QSpinBox()
        self.spinLevel = spin
        spin.setValue(24)
        spin.valueChanged.connect(lambda : self.updateEncounterSummary()) 
        # Don't focus just on mouse wheel
        spin.setFocusPolicy(Qt.StrongFocus)
        hbox.addWidget(spin)
        hbox.addStretch()
        button = QPushButton("Clear")
        button.clicked.connect(lambda : self.encounterTable.setRowCount(0))
        hbox.addWidget(button)
        # XXX Need to disable this button when there are no rows in the
        #     encounter table
        button = QPushButton("Add To Scene")
        self.addTokensButton = button
        hbox.addWidget(button)
        
        hbox.setStretchFactor(label, 0)
        hbox.setStretchFactor(spin, 0)

        label = QLabel("Summary")
        self.summaryLabel = label

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(hbox)
        
        vbox.addWidget(table)
        vbox.addWidget(label)
        widget = QWidget()
        widget.setLayout(vbox)
        vsplitter.addWidget(widget)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(vsplitter)
        
        self.setLayout(vbox)

    def addMonsterToEncounter(self, row):
        logger.info("row %d", row)
        table = self.encounterTable

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        # Add to the encounter table
        nRow = table.rowCount()
        table.setRowCount(nRow + 1)
        for i in xrange(table.columnCount()):
            header = table.horizontalHeaderItem(i).text()
            iMonster = self.monsterHeaderToColumn.get(header, None)
            # Some columns in the encounter don't have a monster counterpart,
            # ignore
            cell = ""
            if (iMonster is not None):
                cell = self.monsterTable.item(row, iMonster).text()

            elif (i == self.encounterHeaderToColumn["HP"]):
                iHDD = self.monsterHeaderToColumn["HDD"]
                hdd = int(self.monsterTable.item(row, iHDD).text())
                iHDB = self.monsterHeaderToColumn["HDB"]
                hdb = int(self.monsterTable.item(row, iHDB).text())
                iHD = self.monsterHeaderToColumn["HD"]
                hd = int(self.monsterTable.item(row, iHD).text())
                hp = sum([random.randint(1, hdd) for _ in xrange(hd)]) + hdb
                cell = "%d" % hp
                
            elif (i == self.encounterHeaderToColumn["Id"]):
                cell = self.monsterTable.item(row, self.monsterHeaderToColumn["Name"]).text()
                
            # Add at the end of the encounter table
            if (i == self.encounterHeaderToColumn["Name"]):
                link = self.monsterTable.item(row, self.monsterHeaderToColumn["Link"]).text()
                item = LinkTableWidgetItem(cell, link)
            else:
                item = QTableWidgetItem(cell)
            table.setItem(nRow, i, item)

        table.setCurrentItem(item)

        table.setSortingEnabled(True)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        self.updateEncounterSummary()

    def updateEncounterSummary(self):
        numMonsters = self.encounterTable.rowCount()
        hitDice = 0
        hp = 0
        xp = 0
        attacksRound = 0
        for j in xrange(self.encounterTable.rowCount()):
            hitDice += int(self.encounterTable.item(j, self.encounterHeaderToColumn["HD"]).text())
            xp += int(self.encounterTable.item(j, self.encounterHeaderToColumn["XP"]).text())
            hp += int(self.encounterTable.item(j, self.encounterHeaderToColumn["HP"]).text())
            # XXX This should match numbers, some attacks have text
            try:
                attacksRound += int(self.encounterTable.item(j, self.encounterHeaderToColumn["#AT"]).text())
            except:
                pass

        levelSum = self.spinLevel.value()
        if (hitDice < levelSum * 0.75):
            rating = "EASY"

        elif (hitDice < levelSum * 1.10):
            rating = "MEDIUM"

        elif (hitDice < levelSum * 1.25):
            rating = "HARD"

        elif (hitDice < levelSum * 2.0):
            rating = "IMPOSSIBLE"

        else:
            rating = "DIE DIE DIE!!!"
        
        # 6 monsters, 24 Hit Dice, 2430 XP, 6 attacks/round, IMPOSSIBLE
        self.summaryLabel.setText("%d monster%s, %d Hit Dice, %0.1f HP/monster, %d XP, %d attacks/round, %s" % 
            (numMonsters, "s" if numMonsters > 1 else "", hitDice, float(hp) / numMonsters, xp, attacksRound, rating))


    def eventFilter(self, source, event):
        logger.debug("source %s type %s", class_name(source), qEventTypeToString(event.type()))
        if ((source == self.encounterTable) and (event.type() == QEvent.ShortcutOverride) and
            (event.key() == Qt.Key_Delete)):
            logger.info("%d", event.key())
            # XXX Ignore the global "del" key shortcut so cells can be edited,
            #     fix in some other way (per dockwidget actions?) and remove, 
            #     see Qt.ShortcutContext
            event.accept()
            return True

        elif ((source == self.encounterTable) and (event.type() == QEvent.KeyPress) and 
            (event.key() == Qt.Key_Delete) and (self.encounterTable.currentItem() is not None)):
            logger.info("Deleting encounter table row %d", self.encounterTable.currentItem().row())
            self.encounterTable.removeRow(self.encounterTable.currentItem().row())
            self.updateEncounterSummary()
            return True

        elif ((source == self.monsterTable) and (event.type() == QEvent.KeyPress) and 
            (event.key() == Qt.Key_Return) and (self.monsterTable.currentItem() is not None)):
            logger.info("Adding monster to encounter %d", self.monsterTable.currentItem().row())

            self.addMonsterToEncounter(self.monsterTable.currentItem().row())
            return True

        return False

    def filterItems(self, filter):
        if (filter is not None):
            filter = filter.lower()
            words = filter.split()
        for j in xrange(self.monsterTable.rowCount()):
            if (filter is None):
                self.monsterTable.showRow(j)
            
            else:
                for i in xrange(self.monsterTable.columnCount()):
                    item = self.monsterTable.item(j, i)
                    if (all([word in item.text().lower() for word in words])):
                        self.monsterTable.showRow(j)
                        break

                else:
                    self.monsterTable.hideRow(j)
                
class DocBrowser(QWidget):
    """
    Documentation browser with HTML browser, filtering/searching, filter/search
    results list, filter/search results list and hit navigation and table of
    contents tree
    """
    index = None
    docDirpath = R"..\..\jscript\vtt\_out"
    def __init__(self, parent=None):
        super(DocBrowser, self).__init__(parent)

        # XXX Note Qt QTextEdit supports markdown format since 5.14 but the
        #     installed version is 5.3.1. Would need to roll own markdown to
        #     rich text editor (see
        #     https://doc.qt.io/qt-6/richtext-structure.html for tables, etc) or
        #     to html (note plain markdown doesn't support tables, so ideally
        #     would need to be a mixed markdown/html editor?)
        #     Could also use a webview with markdown.js/marked.js or such
        #     Could also use some python markdown package to translate to richtext
        #     or html

        # XXX Allow page bookmarks
        # XXX Allow query bookmarks eg title:monstrous to search a monster,
        #     title:wizard title:spell to search for a wizard spell
        # XXX Allow hiding the query and result list

        # XXX Missing sync listitem when navigating if possible?

        if (self.index is None):
            indexFilepath = os.path.join("_out", "index.json")
            if (not os.path.exists(indexFilepath)):
                build_index(self.docDirpath)

            with open(indexFilepath, "r") as f:
                js = json.load(f)

            index = Struct()
            index.word_to_filepaths = js["word_to_filepaths"]
            index.title_to_filepath = js["title_to_filepath"]
            index.filepath_to_title = js["filepath_to_title"]
            
            # Convert from dict of lists to dict of sets
            index.word_to_filepaths = { key: set(l) for key, l in index.word_to_filepaths.iteritems() }

            self.index = index

        lineEdit = QLineEdit(self)
        lineEdit.setPlaceholderText("Search...")
        listWidget = QListWidget()
        tocTree = QTreeWidget()
        tocTree.setColumnCount(1)
        tocTree.setHeaderHidden(True)
        
        self.lastCursor = None
        self.lastPattern = None
        self.curTocFilePath = None
        self.curTocItem = None
        self.sourceTitle = ""
        
        # XXX Have browser zoom, next, prev buttons / keys
    
        textEdit = QTextBrowser()

        # Don't focus these on wheel scroll
        textEdit.setFocusPolicy(Qt.StrongFocus)
        listWidget.setFocusPolicy(Qt.StrongFocus)
        self.textEdit = textEdit
        self.lineEdit = lineEdit
        self.listWidget = listWidget
        self.tocTree = tocTree
        

        lineEdit.textChanged.connect(self.textChanged)
        lineEdit.returnPressed.connect(self.returnPressed)
        listWidget.currentItemChanged.connect(self.listCurrentItemChanged)
        textEdit.sourceChanged.connect(self.browserSourceChanged)
        tocTree.currentItemChanged.connect(self.treeCurrentItemChanged)
        
        test_font = True
        if (test_font):
            # XXX Temporary font mocking, use CSS or such
            fontId = QFontDatabase.addApplicationFont(os.path.join("_out", "fonts", "Raleway-Regular.ttf"))
            logger.info("Font Families %s",QFontDatabase.applicationFontFamilies(fontId))
            font = QFont("Raleway")
            font.setPointSize(10)
            #font.setWeight(QFont.Bold)
            textEdit.setFont(font)

        # QTextEdit can display html but for readonly browsing it's easier to
        # use QTextBrowser since it has navigation implemented out of the box
        # XXX In theory TextBrowser should already be using
        #     TextBrowserInteraction which contains LinksAccessibleXXX ?
        textEdit.setTextInteractionFlags(textEdit.textInteractionFlags() | Qt.LinksAccessibleByMouse)

        # When setting the html text manually, the images and links can be
        # resolved by either replacing them (with or without file: protocol) or
        # using setSearchPaths to the absolute or relative path where the html
        # comes from
        #html = open("_out/html/dragcred.html", "r").read()
        # This works
        #textEdit.setSearchPaths([R"_out\html"]) 
        # Setting setMetaInformation with an absolute or relative file: url,
        # works only for images on the first page, but then the other urls in the
        # page don't work
        #textEdit.document().setMetaInformation(QTextDocument.DocumentUrl, "file:_out/html/dragcred.html")
        #textEdit.setHtml(html)

        # Using setSource works with both images and other urls, no need to do
        # anything else
        #textEdit.setSource(QUrl.fromLocalFile(R"..\..\jscript\vtt\_out\mm\dragcred.html"))
        
        
        hsplitter = QSplitter(Qt.Horizontal)
        self.hsplitter = hsplitter
        hsplitter.addWidget(tocTree)
        hsplitter.addWidget(textEdit)
        hsplitter.setStretchFactor(1, 20)
        
        vsplitter = QSplitter(Qt.Vertical)
        self.vsplitter = vsplitter
        vsplitter.addWidget(listWidget)
        vsplitter.addWidget(hsplitter)
        vsplitter.setStretchFactor(1, 10)

        hbox = QHBoxLayout()
        label = QLabel("Filter")
        hbox.addWidget(label)
        hbox.addWidget(lineEdit)
        hbox.setStretchFactor(label, 0)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(hbox, 0)
        vbox.addWidget(vsplitter)

        self.setLayout(vbox)

    def saveState(self):
        logger.info("%s", self.getSourcePath())
        
        data = QByteArray()
        stream = QDataStream(data, QBuffer.WriteOnly)
        stream.writeBytes(self.getSourcePath())
        stream.writeBytes(self.vsplitter.saveState())
        stream.writeBytes(self.hsplitter.saveState())
        stream.writeBytes(self.lineEdit.text())

        return data

    def restoreState(self, data):
        logger.info("")
        stream = QDataStream(data, QBuffer.ReadOnly)
        filepath = stream.readBytes()
        self.vsplitter.restoreState(stream.readBytes())
        self.hsplitter.restoreState(stream.readBytes())
        self.lineEdit.setText(stream.readBytes())
        # Empty browsers store "" as path and setSourcePath errors with
        # that, skip
        if (filepath != ""):
            self.setSourcePath(filepath)

    def setSourcePath(self, filepath):
        logger.info("%r", filepath)
        url = QUrl.fromLocalFile(filepath)
        # Update the title before setting the source so it's ready for
        # sourceChanged handlers to fetch it
        # XXX This is broken when filepath came from navigating via links as 
        #     opposed to filtering, /C:/Users/ ... is stored instead of ../../jscript ...
        #     fix in getSourcePath
        self.sourceTitle = self.index.filepath_to_title[os.path.relpath(filepath, self.docDirpath).lower()]
        self.textEdit.setSource(url)

    def getSourcePath(self):
        return self.textEdit.source().path()

    def getSourceTitle(self):
        return self.sourceTitle

    def returnPressed(self):
        logger.info("returnPressed modifiers 0x%x ctrl 0x%x shift 0x%x ", 
            int(qApp.keyboardModifiers()),
            int(qApp.keyboardModifiers()) & Qt.ControlModifier,
            int(qApp.keyboardModifiers()) & Qt.ShiftModifier
        )
        # PyQt complains if 0 is passed as findFlags to find, use
        # FindCaseSensitively for no flags since QRegexp search ignores that
        # anyway
        findFlags = QTextDocument.FindBackward if ((int(qApp.keyboardModifiers()) & Qt.ShiftModifier) != 0) else QTextDocument.FindFlag(0)

        # Yellow the previously current position
        fmt = self.lastCursor.charFormat()
        fmt.setBackground(Qt.yellow)
        self.lastCursor.setCharFormat(fmt)
        self.textEdit.setTextCursor(self.lastCursor)

        if (((int(qApp.keyboardModifiers()) & Qt.ControlModifier) != 0) or 
            (not self.textEdit.find(QRegExp(self.lastPattern, Qt.CaseInsensitive), findFlags))):
            logger.info("findFlags %d", findFlags)
            delta = 1
            if (findFlags == QTextDocument.FindBackward):
                delta = -1
            logger.info("Going to %d item", delta)
            # XXX If there's only one item this fails to trigger the signal
            #     and restart the search, on one hand this tells the user the
            #     search is over instead of cycling the page over and over, 
            #     but with two files it would be cycling through both?
            self.listWidget.setCurrentRow((self.listWidget.currentRow() + delta) % self.listWidget.count())

        else:
            # Orange the next find
            textCursor = self.textEdit.textCursor()
            self.lastCursor = QTextCursor(textCursor)
            fmt = textCursor.charFormat()
            fmt.setBackground(QColor("orange"))
            textCursor.setCharFormat(fmt)
            self.textEdit.setTextCursor(textCursor)

            textCursor.clearSelection()
            self.textEdit.setTextCursor(textCursor)

    def textChanged(self, s):
        
        commonResults = set()
        # Another option is to use a QCompleter, which works but only has three
        # kinds of search: prefix, suffix and contains, doesn't support exact
        # word search which is normally the most accurate result if you know
        # what you are looking for
        for word in s.split():
            logger.info("Matching word %r", word)
            
            # XXX Ideally should first match title by exact word, then
            #     title by prefix, then body by exact word, then body by
            #     prefix, then body by contains, until max matches are
            #     filled, how would that affect findprev/next navigation?

            if (word.startswith("title:")):
                # XXX Have other keywords or shortcuts (eg PHB) and a
                #     language for filtering eg title:hobgoblin,
                #     toc:spell (search all tocs), dir:PHB (search in
                #     the phb dir), terrain:arctic etc for MM, ! for negating
                #     filter?

                # XXX For matching multi word titles one title:prefix is
                #     needed per word, use title:"a b c"  or use title:
                #     for all the words until end of line?
                
                # XXX Do substring match if not enough hits, but always
                #     sort exact matches first? How does that affect
                #     findnext/prev navigation?
                word = word[len("title:"):]
                results = set()
                # XXX Do a reverse hash for title to filepath
                for filepath, title in self.index.filepath_to_title.iteritems():
                    if (re.search(r"\b%s\b" % word, title, re.IGNORECASE) is not None):
                        results.add(filepath)

            else:
                # Find in body
                results = self.index.word_to_filepaths.get(word.lower(), commonResults)
            
            logger.info("Word %r matched to %s", word, results)

            if (len(results) == 0):
                # No matches, further intersections will be empty, bail out
                break

            if (len(commonResults) > 0):
                commonResults = commonResults & results
                    
            else:
                commonResults = results

        items = []
        i = 0
        max_hits = 50
        
        for filepath in commonResults:
            title = self.index.filepath_to_title[filepath]
            #logger.info(title)
            items.append("%s" % title)
            i += 1
            if (i > max_hits):
                break

        items = sorted(items)

        # Remove all rows
        self.listWidget.clear()
        if (len(items) == 0):
            # Empty results, reload the current document so marks are cleared
            self.textEdit.reload()

        else:
            self.listWidget.addItems(items)
            logger.info("Going to %d item", 0)
            self.listWidget.setCurrentRow(0)

    def treeCurrentItemChanged(self, current, previous):
        if (current is None):
            return

        filepath = current.data(0, Qt.UserRole)
        filepath = os_path_normall(filepath)
        urlpath = os_path_normall(self.textEdit.source().path())

        logger.info("Sync browser to tree %s", filepath)
        if (urlpath != filepath):
            self.setSourcePath(filepath)
            
    def browserSourceChanged(self, url):
        logger.info("browserSourceChanged %r", url.path())
        filepath = os_path_normall(url.path())
            
        # XXX Preprocess the toc into a json or such
        tocFilepath = None
        if (os_path_normall("cdrom/WEBHELP/PHB") in filepath):
            # The different levels are expressed as font sizes 4 and 3
            tocFilepath = "cdrom\WEBHELP\PHB\DD01405.HTM"

        elif (os_path_normall("cdrom/WEBHELP/DMG") in filepath):
            tocFilepath = "cdrom\WEBHELP\DMG\DD00183.HTM"
            # The different levels are expressed as font sizes 4 and 3            

        else:
            # XXX Use MM00000.htm for monsters (alphabetic index) 
            logger.info("Hiding toc")
            self.tocTree.hide()
            self.curTocFilePath = None

        if (tocFilepath is not None):
            logger.info("Showing toc")
            self.tocTree.show()
            if (tocFilepath != self.curTocFilePath):
                self.curTocFilePath = tocFilepath
                # Generate toc and set on tree
                self.tocTree.clear()
                logger.info("Generating toc for %r", tocFilepath)
                with open(os.path.join(self.docDirpath, tocFilepath), "r") as f:
                    # XXX Instead of regexp parsing the html, another option
                    #     would be to parse with Qt and detect font sizes,
                    #     but it doesn't feel any better
                    tocText = f.read()
                    curItem = None
                    curFontSize = None
                    for m in re.finditer(r'<FONT [^>]*SIZE="(\d+)[^>"]*">', tocText):
                        fontSize = int(m.group(1))
                        tocEntryText = tocText[m.start():]
                        # find the font end
                        m = re.search(r'</FONT>', tocEntryText)
                        tocEntryText = tocEntryText[:m.end()]
                        # find the anchor, remove any fragments
                        m = re.search(r'<A HREF="([^#"]*)(?:#[^"]*)?">([^<]+)</A>', tocEntryText)
                        if (m is None):
                            # There are some headings in the toc, those are not
                            # really toc entries, in addition, some entry names
                            # are empty (they replicate the next one with an
                            # empty text), don't match those
                            continue
                        entryHref = m.group(1)
                        entryName = m.group(2)

                        logger.debug("Found toc entry %d - %r", fontSize, entryName)
                        
                        if (curItem is None):
                            # XXX Create the dummy entry elsewhere? 
                            # XXX The toc tree could also show all the
                            #     registered tocs at the top level?
                            parentItem = QTreeWidgetItem(["Preface"])
                            self.tocTree.addTopLevelItem(parentItem)
                            parentItem.setData(0, Qt.UserRole, tocFilepath)
                            
                            curFontSize = fontSize
                            curItem = QTreeWidgetItem([entryName])
                            parentItem.addChild(curItem)

                        elif (fontSize == curFontSize):
                            parentItem = curItem.parent()
                            curItem = QTreeWidgetItem([entryName])
                            if (parentItem is not None):
                                parentItem.addChild(curItem)
                            else:
                                self.tocTree.addTopLevelItem(curItem)

                        elif (fontSize > curFontSize):
                            # Go to the parent, add to it
                            # XXX Assumes contiguous font sizes
                            parentItem = curItem
                            for _ in xrange(fontSize - curFontSize + 1):
                                parentItem = parentItem.parent()
                            curItem = QTreeWidgetItem([entryName])
                            if (parentItem is not None):
                                parentItem.addChild(curItem)
                            else:
                                self.tocTree.addTopLevelItem(curItem)
                            curFontSize = fontSize
                            
                        elif (fontSize < curFontSize):
                            # Shouldn't skip levels when nesting
                            assert (curFontSize - fontSize) == 1
                            parentItem = curItem
                            curItem = QTreeWidgetItem([entryName])
                            curFontSize = fontSize
                            parentItem.addChild(curItem)

                        curItem.setData(0, Qt.UserRole, os.path.join(os.path.dirname(filepath), entryHref))

            # Activate the item for this filepath
            itemStack = [self.tocTree.topLevelItem(0)]
            nfilepath = os_path_normall(filepath)
            while (len(itemStack) > 0):
                item = itemStack.pop()
                # Note hrefs case may mismatch from filepaths case and from
                # XXX Fix outside of the loop?

                nitempath = os_path_normall(item.data(0, Qt.UserRole))
                if (nfilepath == nitempath):
                    logger.info("Found item match %r", nitempath)
                    self.tocTree.setCurrentItem(item)
                    break
                # Push the next sibling
                if (item.parent() is not None): 
                    i = item.parent().indexOfChild(item)
                    if ((i >= 0) and (i + 1 < item.parent().childCount())):
                        itemStack.append(item.parent().child(i + 1))
                else:
                    i = self.tocTree.indexOfTopLevelItem(item)
                    if ((i >= 0) and (i + 1 < self.tocTree.topLevelItemCount())):
                        itemStack.append(self.tocTree.topLevelItem(i+1))
                # Push the first child
                if (item.childCount() > 0):
                    itemStack.append(item.child(0))
                

        # We could also show the outline for the browsed html file but this
        # is not useful for the currently indexed files since they have a
        # very simple structure 
        
    def listCurrentItemChanged(self, current, previous):

        if (current is None):
            # The list became empty, nothing to do 
            return

        # Load the new file into the browser
        l = self.index.filepath_to_title.values()
        i = index_of(l, current.text())
        filepath = self.index.filepath_to_title.keys()[i]

        self.setSourcePath(os.path.join(self.docDirpath, filepath))

        # Highlight the search string removing any leading "title:" if
        # present
        # XXX Lineedit should probably export the list of words being
        #     searched without title:
        pattern = str.join("|", [r"\b%s\b" % w[max(0, w.find("title:")):] for w in self.lineEdit.text().split()])
        while (self.textEdit.find(QRegExp(pattern, Qt.CaseInsensitive))):
            cursor = self.textEdit.textCursor()
            fmt = cursor.charFormat()
            fmt.setBackground(Qt.yellow)
            cursor.setCharFormat(fmt)

        # Note it's possible no hits were found in the body if the word was
        # found in the title

        # Go to the first finding orange it and clear the selection
        textCursor = self.textEdit.textCursor()
        textCursor.movePosition(QTextCursor.Start)
        self.textEdit.setTextCursor(textCursor)
        self.textEdit.find(QRegExp(pattern, Qt.CaseInsensitive))

        # Orange the first finding
        textCursor = self.textEdit.textCursor()
        self.lastCursor = QTextCursor(textCursor)
        self.lastPattern = pattern

        fmt = textCursor.charFormat()
        fmt.setBackground(QColor("orange"))
        textCursor.setCharFormat(fmt)
        self.textEdit.setTextCursor(textCursor)

        textCursor.clearSelection()
        self.textEdit.setTextCursor(textCursor)
    
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

            if (img_bytes is not None):
                logger.debug("wrapping in bytes")
                f = io.BytesIO(img_bytes)
                clength = len(img_bytes)

            else:
                # This can happen at application startup when img_bytes is not
                # ready yet
                # XXX Fix in some other way?
                logger.debug("null img_bytes, sending empty")
                f = io.BytesIO("")
                clength = 0

            logger.debug("returning")
            ctype = imctype

        elif (self.path == "/fog.svg"):
            ctype = "image/svg+xml"
            clength = os.path.getsize("_out/fog.svg")

            f = open("_out/fog.svg", "rb")

        elif (self.path == "/index.html"):
            # XXX This neesd updating to use fog.svg instead of image.png if
            #     using svg
            ctype = "text/html"
            clength = os.path.getsize("index.html")

            f = open("index.html", "rb")

        elif (self.path.startswith("/token.png")):
            ctype = "image/png"

            # XXX This is just a mock up, at the very least should send more
            #     compressed jpegs
            if (self.path.endswith("?id=1")):
                filename = "Female_Elf_Warrior_T02.png"
            
            else:
                filename = "Female_Human_Wizard_T01.png"
            
            filepath = os.path.join("_out", "tokens", filename)
            clength = os.path.getsize(filepath)

            f = open(filepath, "rb")

        else:
            # Can't use super since BaseRequestHandler doesn't derive from object
            return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

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
        # Can't use super since BaseRequestHandler doesn't derive from object
        return SimpleHTTPServer.SimpleHTTPRequestHandler.do_HEAD(self)


def server_thread(arg):
    server_address = ("", 8000)
    handler_class = VTTHTTPRequestHandler
    httpd = SocketServer.TCPServer(tuple(server_address), handler_class)

    import socket
    global server_ips
    server_ips = socket.gethostbyname_ex(socket.gethostname())
    
    # XXX In case some computation is needed between requests, this can also do
    #     httpd.handle_request in a loop
    httpd.serve_forever()

server_ips = []
most_recently_used_max_count = 10

class ImageWidget(QScrollArea):
    """
    See https://code.qt.io/cgit/qt/qtbase.git/tree/examples/widgets/widgets/imageviewer/imageviewer.cpp
    
    XXX Note that approach uses a lot of memory because zooming creates a background
        pixmap of the zoomed size. Use a GraphicsView instead? 
        implement scrollbar event handling and do viewport-only zooming?
    """
    def __init__(self, parent=None):
        logger.info("")

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
        self.setStyleSheet("background-color:rgb(196, 196, 196);")

        self.viewport().installEventFilter(self)

        self.setCursor(Qt.OpenHandCursor)

    def setImage(image):
        if (image.colorSpace().isValid()):
            image.convertToColorSpace(QColorSpace.SRgb)
        self.setPixmap(QPixmap.fromImage(image))

    def setPixmap(self, pixmap):
        logger.info("setPixmap %s", pixmap.size())
        self.imageLabel.setPixmap(pixmap)
        # Note this doesn't update the widget, the caller should call some
        # resizing function to update
        
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
        logger.debug("source %s type %s", class_name(source), qEventTypeToString(event.type()))
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


def load_test_scene():
    ds_filepath = "hidden hold.dsz"
    map_filepath = "hidden hold_42x29.png"
    img_offset_in_cells = (14, 13)

    #ds_filepath = "map.ds"
    #map_filepath = "map_36x25.png"
    #img_offset_in_cells = (-2, -4)
    
    #ds_filepath = os.path.join("fonda.ds")
    #map_filepath = os.path.join("fonda_31x25.png")
    #img_offset_in_cells = (-8, -0)


    ds_filepath = "Prison of Sky.ds"
    map_filepath = "Prison of Sky_56x39.png"
    img_offset_in_cells = (27, 35)

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
    #map_filepath = os.path.join("maps", "[OC][Art] Path Among The Clouds Battle Map 25x25.jpg")
    #map_filepath = os.path.join("maps","Straight Forest Road Battle Map 22x30.png")
    #img_offset_in_cells = (-2, 11)

    ds_filepath = os.path.join("_out", ds_filepath)
    map_filepath = os.path.join("_out", map_filepath)

    scene = load_ds(ds_filepath, map_filepath, img_offset_in_cells)

    return scene


class GraphicsPixmapNotifyItem(QGraphicsPixmapItem):
    def sceneEventFilter(self, watched, event):
        # The main token item receives events from the label item in order to
        # detect when label editing ends
        logger.debug("watched %s type %s", class_name(watched), qEventTypeToString(event.type()))

        # XXX Would like to focus on enter or double click and ignore single
        #     click, but events received here are the synthetic
        #     GraphicsSceneMousePress, which is generated after focusin so it's
        #     too late to override, maybe could be filtered elsewhere in the
        #     original item or the view?
        
        # if (event.type() == QEvent.GraphicsSceneMousePress):
        #     logger.info("Skipping mouse press")
        #     return True
        # elif (event.type() == QEvent.GraphicsSceneMouseDoubleClick):
        #     logger.info("Skipping double mouse press")
        #     watched.setFocus(Qt.MouseFocusReason)
        #     return True
        handled = False
        gscene = self.scene()
        if (event.type() == QEvent.FocusOut):
            # Recenter the text, update map_token
            gscene.adjustTokenGeometry(self)
            watched.setTextInteractionFlags(Qt.NoTextInteraction)
            self.data(0).name = watched.toPlainText()

            # Make the scene dirty which propagates the change to the image and
            # to the tree
            gscene.makeDirty()

        elif ((event.type() == QEvent.KeyPress) and watched.hasFocus()):
            # Recenter on every keypress
            
            # XXX Note this is not exhaustive (could use the mouse or other to
            #     insert text in the label), would like an onchange notification
            #     but GraphicsTextItems don't have that.
            if (event.key() == Qt.Key_Escape):
                # Restore original value and focus the parent to end editing
                # mode
                gscene.setTokenLabelText(self, self.data(0).name)
                self.setFocus()
                result = True

            elif (event.key() == Qt.Key_Return):
                # Setting the focus to the parent causes focus out and accept
                self.setFocus()

            else:
                result = watched.sceneEvent(event)

            gscene.adjustTokenGeometry(self)
            handled = True
            
        if (not handled):
            result = watched.sceneEvent(event)

        return result

    def boundingRect(self):
        """
        Nested QGraphicsItems don't return the children + parent boundingRect by
        default, just the parent. This causes corruption "trails" when dragging
        the parent if the children paint outside the parent.
        
        Return parent + children boundingRect, but this has issues with code
        using the bounding rect expecting to get only the pixmap bounding rect
        (eg centering is affected by the label size)
        """
        
        rect = super(GraphicsPixmapNotifyItem, self).boundingRect()
        childRect = self.childrenBoundingRect()
        if (len(self.childItems()) > 0):
            if (False):
                logger.debug("%s vs %s + %s (%s) = %s", qtuple(rect), 
                    qtuple(self.pixmap().rect()), 
                    qtuple(self.childItems()[0].boundingRect()), 
                    qtuple(self.childrenBoundingRect()), 
                    qtuple(rect | self.childrenBoundingRect()))
            
            childRect.moveTo(self.childItems()[0].pos())
        return rect | childRect

    def itemChange(self, change, value):
        logger.debug("")
        # Scene can be none before the item is added to the scene
        if (self.scene() is not None):
            return self.scene().itemChanged(self, change, value)

        else:
            return super(GraphicsPixmapNotifyItem, self).itemChange(change, value)

class VTTGraphicsScene(QGraphicsScene):
    def __init__(self, map_scene, parent = None):
        super(VTTGraphicsScene, self).__init__(parent)

        self.tokenItems = set()
        self.doorItems = set()
        self.openDoorItems = set()
        self.wallItems = set()
        self.imageItems = set()
        self.map_scene = map_scene

        self.blendFog = False
        self.fogVisible = False
        self.lockDirtyCount = False
        self.dirtyCount = 0
        self.fogPolysDirtyCount = -1

        self.allWallsItem = QGraphicsItemGroup()
        self.allDoorsItem = QGraphicsItemGroup()
        self.fogItem = QGraphicsItemGroup()
        self.gridItem = QGraphicsItemGroup()
        self.fogCenter = None
        self.fogCenterLocked = False

        self.cellDiameter = 70
        self.snapToGrid = True
        self.gridVisible = True
        self.lightRange = 0.0

        self.addItem(self.allWallsItem)
        self.allWallsItem.setZValue(0.1)
        self.addItem(self.allDoorsItem)
        self.allDoorsItem.setZValue(0.2)
        self.addItem(self.fogItem)
        self.fogItem.setZValue(0.9)
        # XXX Tabulate z values (eg tokens should be over doors, walls and grids?)
        self.addItem(self.gridItem)
        self.gridItem.setZValue(0.3)

    def makeDirty(self):
        # Ignore dirty calls from inside fog updating to prevent infinite
        # recursion
        # XXX Not clear this really works because scene changes are batched
        if (not self.lockDirtyCount):
            self.dirtyCount += 1

    def setLockDirtyCount(self, locked):
        self.lockDirtyCount = locked

    def itemChanged(self, item, change, value):
        """
        This receives notifications from GraphicsPixmapNotifyItem
        """
        logger.info("change %s", change)
        
        if ((change == QGraphicsItem.ItemPositionChange) and self.snapToGrid):
            # value is the new position, snap 
            snap_granularity = self.cellDiameter / 2.0
            snap_pos = (value / snap_granularity).toPoint() * snap_granularity

            return snap_pos
        
        if (change == QGraphicsItem.ItemPositionHasChanged):
            item.data(0).scene_pos = qtuple(item.scenePos())

        if (change in [
            QGraphicsItem.ItemVisibleHasChanged, 
            QGraphicsItem.ItemPositionHasChanged,
            # Need to update fog if the focus changes
            QGraphicsItem.ItemSelectedHasChanged,
            ]):
            if (not self.lockDirtyCount):
                self.makeDirty()
                if (self.getFogCenter() is not None):
                    self.updateFog()
        
        return value

    def getFogCenter(self):
        logger.info("fogCenter %s", self.selectedItems())
        # Note focusItem() is None when the view tab focus is switched away
        # unless stickyFocus is set, and selectedItems is always empty unless
        # items are selected. Also, focusItem() can be any item that can be
        # focused (eg tokens or doors), only use tokens as fog centers
        focusItem = self.focusItem()
        if ((self.isToken(focusItem)) and (not self.fogCenterLocked)):
            # The fogcenter is a token and tokens are centered on 0,0, no need
            # to adjust pos() in any way
            self.fogCenter = focusItem.pos()
        
        return self.fogCenter

    def setFogCenter(self, fogCenter):
        # This only makes sense if the center is locked, since otherwise will be
        # overwritten as soon as the focusitem is moved
        self.fogCenter = fogCenter

    def setFogCenterLocked(self, locked):
        self.fogCenterLocked = locked
        
    def getFogCenterLocked(self):
        return self.fogCenterLocked

    def setSnapToGrid(self, snap):
        self.snapToGrid = snap

    def getSnapToGrid(self):
        return self.snapToGrid
        
    def setCellDiameter(self, cellDiameter):
        self.cellDiameter = cellDiameter

    def getCellDiameter(self):
        return self.cellDiameter

    def isToken(self, item):
        return (item in self.tokenItems)

    def tokens(self):
        return self.tokenItems

    def tokenAt(self, index):
        return list(self.tokenItems)[index]

    def isImage(self, item):
        return (item in self.imageItems)
    
    def images(self):
        return self.imageItems

    def imageAt(self, index):
        return list(self.imageItems)[index]

    def isWall(self, item):
        return (item in self.imageItems)
    
    def walls(self):
        return self.wallItems

    def isDoor(self, item):
        return item in self.doorItems

    def doors(self):
        return self.doorItems

    def openDoors(self):
        return self.openDoorItems

    def isDoorOpen(self, item):
        return (item in self.openDoorItems)

    def setDoorOpen(self, item, open):
        closed_brush = QBrush(Qt.red)
        open_brush = QBrush(Qt.green)
        
        if (open):
            self.openDoorItems.add(item)
            item.setBrush(open_brush)
                        
        else:
            # Avoid key not found exception at door creation time, but still set
            # the color
            if (self.isDoorOpen(item)):
                self.openDoorItems.remove(item)
            item.setBrush(closed_brush)
        item.data(0).open = open
        # No need to call update, setBrush triggers sceneChanged already

    def isTokenHiddenFromPlayer(self, item):
        return item.data(0).hidden
        
    def setTokenHiddenFromPlayer(self, item, hidden):
        logger.debug("setTokenHiddenFromPlayer %s %s", item.data(0).name, hidden)
        if (hidden):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.4)
            item.setGraphicsEffect(effect)

        else:
            item.setGraphicsEffect(None)

        item.data(0).hidden = hidden

    def setWallsVisible(self, visible):
        self.makeDirty()
        self.allWallsItem.setVisible(visible)

    def setTokensHiddenFromPlayerVisible(self, visible):
        logger.info(visible)
        #self.makeDirty()
        for token in self.tokenItems:
            if (token.data(0).hidden):
                # Changing z value makes the token lose focus, push it beind the
                # image instead

                # XXX Do the same thing for the other items? (walls, doors)
                if (visible):
                    token.setZValue(0.4)
                else:
                    token.setZValue(-1)
                    
    def setDoorsVisible(self, visible):
        self.makeDirty()
        self.allDoorsItem.setVisible(visible)

    def adjustTokenGeometry(self, token):
        pixItem = token
        txtItem = pixItem.childItems()[0]
        pix = pixItem.pixmap()
        map_token = pixItem.data(0)

        pixItem.setScale(map_token.scale / pix.width())
        # QGraphicsPixmapItem are not centered by default, do it
        pixItem.setOffset(-qSizeToPoint(pixItem.pixmap().size() / 2.0))
        txtItem.setScale(1.0/pixItem.scale())
        # Calculate the position taking into account the text item reverse
        # scale                
        pos = QPointF(
            0 - txtItem.boundingRect().width() / (2.0 * pixItem.scale()), 
            pixItem.pixmap().height() /2.0 - txtItem.boundingRect().height() / (2.0 * pixItem.scale())
        )
        txtItem.setPos(pos)
        
    def setTokenLabelText(self, token, text):
        logger.debug("%s", text)
        txtItem = self.getTokenLabelItem(token)
        txtItem.setHtml("<div style='background:rgb(255, 255, 255, 128);'>%s</div>" % text)

    def addToken(self, map_token):
        logger.info("addToken %r", map_token.filepath)

        filepath = map_token.filepath
        pix = QPixmap()
        max_token_size = QSize(128, 128)
        if (not pix.load(filepath)):
            logger.error("Error loading pixmap, using placeholder!!!")
            pix = QPixmap(max_token_size)
            pix.fill(QColor(255, 0, 0))
        # Big tokens are noticeably slower to render, use a max size
        logger.debug("Loading and resizing token %r from %s to %s", filepath, pix.size(), max_token_size)
        pix = pix.scaled(max_token_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixItem = GraphicsPixmapNotifyItem(pix)
        pixItem.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsScenePositionChanges)
        # XXX Tooltip could have more stats (THAC0, AC, etc)
        pixItem.setToolTip("<b>" + map_token.name + "</b>" + "<br><b>T0</b>:%s <b>AT</b>:%s <b>D</b>:%s<br><b>AC</b>:%s <b>HP</b>:%s" %(
            map_token.ruleset_info.T0,
            map_token.ruleset_info.AT,
            map_token.ruleset_info.Damage,
            map_token.ruleset_info.AC,
            map_token.ruleset_info.HP if (map_token.ruleset_info.HP_ == "") else map_token.ruleset_info.HP_
        ))
        pixItem.setPos(*map_token.scene_pos)
        pixItem.setData(0, map_token)
        pixItem.setCursor(Qt.SizeAllCursor)
        # Note setting parent zvalue is enough, no need to set txtItem
        pixItem.setZValue(0.4)
        self.setTokenHiddenFromPlayer(pixItem, map_token.hidden)

        # XXX Monkey patching itemChange doesn't work
        #QGraphicsPixmapItem.itemChange = lambda s, c, v: logger.info("itemChange %s", v)
        
        txtItem = QGraphicsTextItem(pixItem)
        # Use HTML since it allows setting the background color
        self.setTokenLabelText(pixItem, map_token.name)
        # Keep the label always at the same size disregarding the token
        # size, because the label is a child of the pixitem it gets affected
        # by it. Also reduce the font a bit
        font = txtItem.font()
        font.setPointSize(txtItem.font().pointSize() *0.75)
        txtItem.setFont(font)
        # XXX This needs to hook on the item's focusOutEvent to unfocus and to
        #     recenter the label and to update the scene tree widget and the
        #     tooltip
        #     Note 
        #     txtItem.document().setDefaultTextOption(QTextOption(Qt.AlignCenter))
        #     does center the text in the width set with setTextWidth, but
        #     the item anchor is still the item position so still needs to be
        #     set to the top left corner, which changes depending on the 
        #     text length.
        #txtItem.setTextInteractionFlags(Qt.TextEditorInteraction)

        self.adjustTokenGeometry(pixItem)
        
        item = pixItem

        self.tokenItems.add(item)
        self.addItem(item)

        txtItem.installSceneEventFilter(pixItem)

        return item

    def getTokenPixmapItem(self, token):
        return token

    def getTokenLabelItem(self, token):
        return token.childItems()[0]

    def addWall(self, map_wall):
        logger.info("Adding wall %s", map_wall)
        pen = QPen(Qt.cyan)
        item = QGraphicsLineItem(*map_wall)
        # Items cannot be selected, moved or focused while inside a group,
        # the group can be selected and focused but not moved
        item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable)
        item.setPen(pen)
        item.setData(0, map_wall)
        self.wallItems.add(item)
        self.allWallsItem.addToGroup(item)
        
    def addDoor(self, map_door):
        logger.info("Adding door %s", map_door.lines)
        pen = QPen(Qt.black)
        
        # Doors have been expanded to individual lines for ease of fog
        # calculation, convert to polyline
        lines = map_door.lines
        points = [QPointF(lines[(i*4) % len(lines)], lines[(i*4+1) % len(lines)]) for i in xrange(len(lines)/4)]
        points.append(QPointF(lines[-2], lines[-1]))
        
        item = QGraphicsPolygonItem(QPolygonF(points))
        item.setData(0, map_door)
        item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable)
        item.setPen(pen)
        self.setDoorOpen(item, map_door.open)
        self.doorItems.add(item)
        self.allDoorsItem.addToGroup(item)

    def addImage(self, map_image):
        logger.debug("Populating image %r", map_image.filepath)
        item = QGraphicsPixmapItem(QPixmap(map_image.filepath))
        item.setPos(*map_image.scene_pos)
        item.setScale(map_image.scale / item.pixmap().width())
        item.setData(0, map_image)
        ##item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)

        self.imageItems.add(item)
        self.addItem(item)

    def addGrid(self, cellDiameter):
        # XXX This is not idempotent, addGrid should only be called once per
        #     scene
        rect = self.itemsBoundingRect()
        pen = QPen(QColor(0, 0, 0, 128))
        x, y = rect.left(), rect.top()
        while (x < rect.right()):
            lineItem = QGraphicsLineItem(x, rect.top(), x, rect.bottom())
            lineItem.setPen(pen)
            self.gridItem.addToGroup(lineItem)
            x += cellDiameter

        while (y < rect.bottom()):
            lineItem = QGraphicsLineItem(rect.left(), y, rect.right(), y)
            lineItem.setPen(pen)
            self.gridItem.addToGroup(lineItem)
            y += cellDiameter

    def setGridVisible(self, visible):
        logger.info("setGridVisible %s", visible)
        self.makeDirty()
        self.gridItem.setVisible(visible)

    def setLightRange(self, lightRange):
        logger.info("setLightRange %s", lightRange)
        self.makeDirty()
        self.invalidate()
        self.lightRange = lightRange

    def getLightRange(self):
        return self.lightRange
        
    def setFogVisible(self, visible):
        logger.info("setFogVisible %s", visible)
        self.fogVisible = visible
        self.updateFog()

    def setBlendFog(self, blend):
        logger.info("setBlendFog %s", blend)
        self.blendFog = blend
        self.updateFog()
        
    def updateFog(self, force=False):
        logger.info("updateFog force %s dirty %s draw %s blend %s", force, self.fogPolysDirtyCount != self.dirtyCount, self.fogVisible, self.blendFog)
        gscene = self
        
        if (gscene.getFogCenter() is None):
            logger.warning("Called updateFog with no token item!!")
            return

        if (self.blendFog):
            fogBrush = QBrush(QColor(0, 0, 255, 125))
            fogPen = QPen(Qt.black)

        else:
            # Match the background color to prevent map leaks
            fogBrush = QBrush(QColor(196, 196, 196))
            fogPen = QPen(QColor(196, 196, 196))
        
        # XXX Use gscene.blockSignals to prevent infinite looping 
        #     and/or updateEnabled(False) to optimize redraws
        if ((gscene.dirtyCount != gscene.fogPolysDirtyCount) or force):
            gscene.fogPolysDirtyCount = gscene.dirtyCount
            
            token_pos = self.getFogCenter()
            token_x, token_y = qtuple(token_pos)
            # XXX This assumes 60ft vision and 5ft per cell, allow configuring
            fog_polys = compute_fog(self.map_scene, (token_x, token_y), self.lightRange, self.sceneRect())
            self.fog_polys = fog_polys

            fogItem = self.fogItem
            logger.info("updateFog %f %f", token_x, token_y)

            gscene.removeItem(fogItem)

            fogItem = QGraphicsItemGroup()
            # XXX This could use degenerate triangles and a single polygon?
            for fog_poly in fog_polys:
                item = QGraphicsPolygonItem(QPolygonF([QPointF(p[0], p[1]) for p in fog_poly]))
                item.setBrush(fogBrush)
                item.setPen(fogPen)
                fogItem.addToGroup(item)

            fogItem.setVisible(self.fogVisible)
            fogItem.setZValue(0.9)

            self.fogItem = fogItem
            gscene.addItem(fogItem)

        # This is a debug option that is normally disabled, do it after the fact
        # without dirty caching
        else:
            self.fogItem.setVisible(self.fogVisible)
            for item in self.fogItem.childItems():
                item.setPen(fogPen)
                item.setBrush(fogBrush)

        
def compute_fog(scene, fog_center, light_range, bounds):
    logger.info("draw fog %d polys pos %s", len(scene.map_walls), fog_center)
    
    token_x, token_y = fog_center
    fog_polys = []
    # Place a fake circle wall to simulate light range 
    # XXX Note the fake circle wall doesn't account for other lights in the 
    #     scene beyond this range
    light_walls = []
    if (light_range != 0.0):
        r = light_range * 1.0
        sides = 16
        x0, y0 = None, None        
        for s in xrange(sides + 1):
            x1, y1 = (
                token_x + r*math.cos(2.0 * s * math.pi / sides), 
                token_y + r*math.sin(2.0 * s * math.pi / sides)
            )
            if (x0 is not None):
                wall = (x0, y0, x1, y1)
                light_walls.append(wall)
                                    
            x0, y0 = x1, y1

    light_range2 = light_range ** 2

    max_min_l = (bounds.width() ** 2 + bounds.height() ** 2)
    for map_item in scene.map_walls + scene.map_doors + light_walls:
        # Doors are structs, walls are lists
        # XXX Merge walls and convert to structs?
        lines = map_item
        is_door = (hasattr(map_item, "lines"))
        if (is_door):
            if (map_item.open):
                continue
            lines = map_item.lines

        for i in xrange(0, len(lines), 4):
            x0, y0, x1, y1 = lines[i:i+4]
            
            # Optimization, ignore polys with both ends outside of the light
            # range plus epsilon
            # XXX This is wrong if the poly intersects the circle, disabled for
            #     now, needs to check the distance from the line to the point
            if (False and (light_range != 0) and ( 
                (((x0 - token_x) ** 2 + (y0 - token_y) ** 2) > light_range2 + 1.0) and
                (((x1 - token_x) ** 2 + (y1 - token_y) ** 2) > light_range2 + 1.0)
                )):
                continue
            
            # Clip the frustum to the scene bounds, this prevents having to use
            # a large frustum which will cause large polygons to be rendered
            # (slow) and doesn't work for degenerate cases like very close to a
            # wall where the frustum is close to 180 degrees
            min_ls = []
            hits = []
            hitmasks = []
            for px, py, vx, vy in [ 
                (x0, y0, x0 - token_x, y0 - token_y),
                (x1, y1, x1 - token_x, y1 - token_y)
            ]:
                # line = p + v * l -> l = -p / v
                # for top edge y=top, py + vy * l = top -> l = (top - py) / vy
                # Keep the shortest positive distance
                min_l = max_min_l
                hitmask = 0
                hit = 0
                if (vy != 0):
                    l = (bounds.top() - py) / (1.0 * vy)
                    hitmask |= ((l>=0) << 0)
                    if ((l < min_l) and (l >= 0)):
                        hit = 0
                        min_l = l
                    
                    l = (bounds.bottom() - py) / (1.0 * vy)
                    hitmask |= ((l>=0) << 2)
                    if ((l < min_l) and (l >= 0)):
                        hit = 2
                        min_l = l
                    
                if (vx != 0):
                    l = (bounds.right() - px) / (1.0 * vx)
                    hitmask |= ((l>=0) << 1)
                    if ((l < min_l) and (l >= 0)):
                        hit = 1
                        min_l = l
                    
                    l = (bounds.left() - px) / (1.0 * vx)
                    hitmask |= ((l>=0) << 3)
                    if ((l < min_l) and (l >= 0)):
                        hit = 3
                        min_l = l
                    
                min_ls.append(min_l)
                hits.append(hit)
                hitmasks.append(hitmask)

            # Frustum with no extra points yet, but clipped to the scene bounds
            frustum = [
                (x0, y0), 
                (x0 + (x0 - token_x) * min_ls[0], y0 + (y0 - token_y) * min_ls[0]),
                (x1 + (x1 - token_x) * min_ls[1], y1 + (y1 - token_y) * min_ls[1]), 
                (x1, y1)
            ]

            if (hits[0] == hits[1]):
                # Both intersections are on the same bound, no need to insert
                # intermediate points
                pass

            else:
                
                mask = (1 << hits[0]) | (1 << hits[1])
                # Frustum intersects adjacent bounds, insert one corner
                if (mask == 0b0011):
                    frustum.insert(2, qtuple(bounds.topRight()))

                elif (mask == 0b0110):
                    frustum.insert(2, qtuple(bounds.bottomRight()))
                
                elif (mask == 0b1100):
                    frustum.insert(2, qtuple(bounds.bottomLeft()))

                elif (mask == 0b1001):
                    frustum.insert(2, qtuple(bounds.topLeft()))

                # Frustum intersects opposite top & bottom bounds, insert two
                # corners
                elif (mask == 0b0101):
                    if (hitmasks[0] & 0b0010):
                        frustum.insert(2, qtuple(bounds.topRight()))
                        frustum.insert(3, qtuple(bounds.bottomRight()))
                    
                    else:
                        frustum.insert(2, qtuple(bounds.topLeft()))
                        frustum.insert(3, qtuple(bounds.bottomLeft()))
                    
                    if (hits[0] == 2):
                        # Reverse direction, swap
                        frustum[2], frustum[3] = frustum[3], frustum[2]

                # Frustum intersects opposite left & right bounds, insert two
                # corners
                elif (mask == 0b1010):
                    if (hitmasks[0] & 0b0001):
                        frustum.insert(2, qtuple(bounds.topRight()))
                        frustum.insert(3, qtuple(bounds.topLeft()))
                    
                    else:
                        frustum.insert(2, qtuple(bounds.bottomRight()))
                        frustum.insert(3, qtuple(bounds.bottomLeft())) 

                    if (hits[0] == 3):
                        # Reverse direction, swap
                        frustum[2], frustum[3] = frustum[3], frustum[2]      

            fog_polys.append(frustum)
                
    return fog_polys

default_ruleset_info = {
    "Alignment": "", 
    "AC": "7", 
    "A_" : "0",
    "HP_": "", 
    "HP": "7", 
    "Damage": "By weapon", 
    "T0": "20", 
    "HDD": "8", 
    "Notes": "", 
    "AT": "1", 
    "MR": "0", 
    "XP": "35", 
    "Id": "Human, Mercenary", 
    "HD": "1", 
    "HDB": "1"
} 

class VTTGraphicsView(QGraphicsView):
    def __init__(self, parent = None):
        super(VTTGraphicsView, self).__init__(parent)

        self.drawWalls = True
        self.blendMapFog = False
        self.drawMapFog = False
        self.drawGrid = True
        self.snapToGrid = True

        self.installEventFilter(self)

    def setScene(self, gscene):

        gscene.installEventFilter(self)
        return super(VTTGraphicsView, self).setScene(gscene)
    
    def mouseDoubleClickEvent(self, event):
        logger.info("mouseDoubleClickEvent")

        gscene = self.scene()
        if (gscene.focusItem() is not None):
            focusItem = gscene.focusItem()

            if (gscene.isDoor(focusItem)):
                gscene.setDoorOpen(focusItem, not gscene.isDoorOpen(focusItem))    
                gscene.makeDirty()
                
            elif (gscene.isToken(focusItem)):
                # XXX Double click to edit size/rotation?
                map_token = gscene.getTokenPixmapItem(focusItem).data(0)
                filepath = map_token.filepath
                # XXX Get supported extensions from Qt
                filepath, _ = QFileDialog.getOpenFileName(self, "Import Token", filepath, "Images (*.png *.jpg *.jpeg *.jfif *.webp)")

                if ((filepath == "") or (filepath is None)):
                    return

                gscene.removeItem(focusItem)
                map_token.filepath = filepath
                focusItem = gscene.addToken(map_token)
                gscene.setFocusItem(focusItem)
                gscene.makeDirty()

        else:
            r = gscene.getCellDiameter() * 1.0
            sides = 25
            x0, y0 = None, None
            scenePos = self.mapToScene(event.pos())
            
            for s in xrange(sides+1):
                x1, y1 = (
                    scenePos.x() + r*math.cos(2.0 * s * math.pi / sides), 
                    scenePos.y() + r*math.sin(2.0 * s * math.pi / sides)
                )
                if (x0 is not None):
                    wall = (x0, y0, x1, y1)
                    # XXX Missing adding to the main scene, this should be
                    #     passing the main scene wall tuple, not this one
                    gscene.addWall(wall)
                    gscene.map_scene.map_walls.append(wall)
                                        
                x0, y0 = x1, y1

            gscene.makeDirty()
                            
    def wheelEvent(self, event):
        logger.debug("wheelEvent 0x%x", event.modifiers())
        
        # Zoom Factor
        zoomFactor = 1.0015 ** event.angleDelta().y()
        
        # Set Anchors
        
        # XXX Empirically AnchorUnderMouse drifts slightly several pixels,
        #     use NoAnchor and do the calculation manually
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        
        pos = self.mapFromGlobal(event.globalPos())
        
        # Save scale and translate back
        oldPos = self.mapToScene(pos)
        self.scale(zoomFactor, zoomFactor)
        newPos = self.mapToScene(pos)
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        
        # Prevent propagation, not enough with returning True below
        event.accept()
            
        #else:
        #    super(VTTGraphicsView, self).mouseMoveEvent(event)

    def event(self, event):
        logger.info("event 0x%x", event.type())
        # Tab key needs to be trapped at the event level since it doesn't get to
        # keyPressEvent
        gscene = self.scene()
        if ((event.type() == QEvent.KeyPress) and (event.key() in [Qt.Key_Tab, Qt.Key_Backtab]) and 
            ((int(event.modifiers()) & Qt.ControlModifier) == 0)):

            tokens = list(gscene.tokens())
            tokenCount = len(tokens)
            if (tokenCount > 0):
                delta = 1 if (event.key() == Qt.Key_Tab) else -1
                # XXX Note token_items is a set, so it doesn't preserve the
                #     order, may not be important since it doesn't matter
                #     there's no strict order between tokens as long as it's
                #     consistent one (ie the order doesn't change between tab
                #     presses as long as no items were added to the token set)
                # XXX Should probably override GraphicsView.focusNextPrevChild
                #     once moving away from filtering?
                
                focusedIndex = index_of(tokens, gscene.focusItem())
                if (focusedIndex == -1):
                    # Focus the first or last item
                    focusedIndex = tokenCount
                else:
                    # Clear the selection rectangle on the old focused item
                    tokens[focusedIndex].setSelected(False)

                focusedIndex = (focusedIndex + delta) % tokenCount
                focusedItem = tokens[focusedIndex]
                
                gscene.setFocusItem(focusedItem)
                # Select so the dashed rectangle is drawn around
                focusedItem.setSelected(True)
                self.ensureVisible(focusedItem, self.width()/4.0, self.height()/4.0)

                # No need to dirty, setFocusItem/setSelected triggers itemchange

                return True

        return super(VTTGraphicsView, self).event(event)

    def keyPressEvent(self, event):
        logger.info("%s", event)

        gscene = self.scene()
        if (gscene.isToken(gscene.focusItem()) and (event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down])):
            d = { Qt.Key_Left : (-1, 0), Qt.Key_Right : (1, 0), Qt.Key_Up : (0, -1), Qt.Key_Down : (0, 1)}
            # Snap to half cell and move in half cell increments
            # XXX Make this configurable
            # XXX This assumes snapping starts at 0,0
            snap_granularity = gscene.getCellDiameter() / 2.0
            move_granularity = snap_granularity
            delta = QPointF(*d[event.key()]) * move_granularity
            focusItem = gscene.focusItem()

            # Note when the token is a group, the children are the one grabbed,
            # not the group, use the focusitem which is always the group

            # snap then move

            # The token item position is offseted so the token appears centered
            # in its position in the map. Take the size of the token into
            # account when calculating the position in the cell
            
            snap_pos = (focusItem.pos() / snap_granularity).toPoint() * snap_granularity
            
            # Snap in case it wasn't snapped before, this will also allow using
            # the arrow keys to snap to the current cell if the movement is
            # rejected below
            
            # Note QSizeF doesn't operate with QPoint, convert to tuple and back
            # to QPoint
            # The center of the cell needs to be the center of the 
            focusItem.setPos(snap_pos)

            # Intersect the path against the existing walls and doors, abort
            # the movement if it crosses one of those

            # Use a bit of slack to avoid getting stuck in the intersection
            # point due to floating point precision issues, don't use too much
            # (eg. 1.5 or more) or it will prevent from walking through tight
            # caves
            
            # XXX Note this doesn't take the size into account, can get tricky
            #     for variable size tokens, since proper intersection
            #     calculation requires capsule-line intersections
            l = QLineF(snap_pos, snap_pos + delta * 1.10)

            # Note walls are lines, not polys so need to check line to line
            # intersection instead of poly to line. Even if that wasn't the
            # case, PolygonF.intersect is new in Qt 5.10 which is not available
            # on this installation, so do only line to line intersection
            i = QLineF.NoIntersection
            for wall_item in gscene.walls():
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
                for door_item in gscene.doors():
                    if (gscene.isDoorOpen(door_item)):
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
                    # Set the token corner so the token is centered on the
                    # position in the map
                    focusItem.setPos(snap_pos + delta)
                    
            self.ensureVisible(focusItem, self.width()/4.0, self.height()/4.0)
            # Note programmatic setpos doesn't trigger itemchange, dirty the 
            # scene so the fog polygons are recalculated on the next updateFog
            # call.
            gscene.makeDirty()

        elif (gscene.isToken(gscene.focusItem()) and (event.key() == Qt.Key_Return)):
            # Enter editing mode on the token label (no need to check for
            # txtItem not focused since the token is focused and the scene focus
            # textitem and token independently)
            # XXX This is replicated across doubleclick, refactor
            # XXX This should select the whole text
            # XXX Should this use the widget .setFocus  or the scene setfocusItem/setSelected?
            txtItem = gscene.getTokenLabelItem(gscene.focusItem())
            txtItem.setTextInteractionFlags(Qt.TextEditorInteraction)
            txtItem.setFocus(Qt.TabFocusReason)

        elif (gscene.isToken(gscene.focusItem()) and (event.text() in ["-", "+"])):
            # Increase/decrease token size, no need to recenter the token on the
            # current position since tokens are always centered on 0,0
            focusItem = gscene.focusItem()
            delta = 1 if event.text() == "+" else -1
            map_token = focusItem.data(0)
            deltaScale = delta * (gscene.getCellDiameter() / 4.0)
            map_token.scale += deltaScale

            gscene.adjustTokenGeometry(focusItem)

            gscene.makeDirty()

        elif (gscene.isToken(gscene.focusItem()) and (event.text() == " ")):
            # Open the adjacent door
            threshold2 = (gscene.getCellDiameter() ** 2.0) * 1.1
            token_center = gscene.focusItem().sceneBoundingRect().center()
            for door_item in gscene.doors():
                door_center = door_item.sceneBoundingRect().center()
                v = (door_center - token_center)
                dist2 = QPointF.dotProduct(v, v)
                logger.info("checking token %s vs. door %s %s vs. %s", token_center, door_center, dist2, threshold2)
                if (dist2 < threshold2):
                    gscene.setDoorOpen(door_item, not gscene.isDoorOpen(door_item))
                    gscene.makeDirty()
                    break

        elif (gscene.isToken(gscene.focusItem()) and (event.text() == "h")):
            # Hide token from player view
            focusItem = gscene.focusItem()
            
            gscene.setTokenHiddenFromPlayer(focusItem, not gscene.isTokenHiddenFromPlayer(focusItem))
            gscene.makeDirty()
            
        # Some single key actions, but don't swallow keys when the label editing
        # is set
        # XXX Eventually remove these and do actions so they can update
        #     statusbars or menus checkboxes
        elif (not isinstance(gscene.focusItem(), QGraphicsTextItem)):

            if (event.text() == "b"):
                # Toggle fog blending on DM View
                self.blendMapFog = not self.blendMapFog
                gscene.setBlendFog(self.blendMapFog)
                # XXX This needs to update the status bar
                
            elif (event.text() == "f"):
                # Toggle fog visibility on DM View
                self.drawMapFog = not self.drawMapFog
                gscene.setFogVisible(self.drawMapFog)
                # XXX This needs to update the status bar

            elif (event.text() == "g"):
                # Toggle grid visibility
                self.drawGrid = not self.drawGrid
                gscene.setGridVisible(self.drawGrid)
                # XXX This needs to update the status bar

            elif (event.text() in ["l", "L"]):
                # Cycle through light ranges
                # No need to sort these, they get sorted below
                d = { 
                    "None" : 0.0, "candle" : 5.0, "torch" : 15.0, 
                    "light spell" : 20.0, "lantern" : 30.0, "campfire" : 35.0,
                    "infravision" : 60.0,
                }
                # XXX Assumes 5ft per cell, lightRange uses scene units
                sortedValues = [i * gscene.getCellDiameter() / 5.0 for i in sorted(d.values())]
                delta = 1 if event.text() == "l" else -1
                lightRangeIndex = index_of(sortedValues, gscene.getLightRange())
                lightRangeIndex = (lightRangeIndex + delta + len(sortedValues)) % len(sortedValues)
                lightRange = sortedValues[lightRangeIndex]
                gscene.setLightRange(lightRange)
                # XXX This needs to update the status bar
                
            elif (event.text() == "s"):
                # Toggle snap to grid
                self.snapToGrid = not self.snapToGrid
                gscene.setSnapToGrid(self.snapToGrid)
                # XXX This needs to update the status bar
                
            elif (event.text() == "w"):
                # Toggle wall visibility on DM View
                self.drawWalls = not self.drawWalls
                # Always draw doors otherwise can't interact with them
                # XXX Have another option? Paint them transparent?
                # XXX Is painting transparent faster than showing and hiding?
                gscene.setWallsVisible(self.drawWalls)
            
        else:
            super(VTTGraphicsView, self).keyPressEvent(event)

            
    def eventFilter(self, source, event):
        logger.debug("source %s type %s", class_name(source), qEventTypeToString(event.type()))

        return super(VTTGraphicsView, self).eventFilter(source, event)


class VTTMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(VTTMainWindow, self).__init__(parent)

        self.campaign_filepath = None
        self.recent_filepaths = []

        self.gscene = None
        self.scene = None

        # XXX scene.changed reentrant flag because updateImage modifies the
        #     scene, probably fix by having two scenes
        self.sceneDirtyCount = 0

        self.createMusicPlayer()

        self.createActions()

        self.createMenus()

        self.createStatus()

        imageWidget = ImageWidget()
        # Allow the widget to receive mouse events
        imageWidget.setMouseTracking(True)
        # Allow the widget to receive keyboard events
        imageWidget.setFocusPolicy(Qt.StrongFocus)
        imageWidget.installEventFilter(self)
        self.imageWidget = imageWidget

        view = VTTGraphicsView()
        self.graphicsView = view
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        
        tree = QTreeWidget()
        self.tree = tree

        for view, title in [
            (self.tree, "Campaign"), 
            # XXX Once multiple scenes/rooms are supported, these two should be dynamic
            (self.imageWidget, "Player View - %s" % (server_ips,) ),
            (self.graphicsView, "DM View"),
        ]:
            self.wrapInDockWidget(view, title)

        textEdit = QTextEdit()
        self.textEdit = textEdit
        test_font = True
        if (test_font):
            # XXX Temporary font mocking, use CSS or such
            fontId = QFontDatabase.addApplicationFont(os.path.join("_out", "fonts", "Raleway-Regular.ttf"))
            logger.info("Font Families %s",QFontDatabase.applicationFontFamilies(fontId))
            font = QFont("Raleway")
            font.setPointSize(10)
            #font.setWeight(QFont.Bold)
            textEdit.setFont(font)
        textEdit.setTextInteractionFlags(textEdit.textInteractionFlags() | Qt.LinksAccessibleByMouse)

        self.setCentralWidget(self.textEdit)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)

        qApp.focusChanged.connect(self.focusChanged)

        #self.centralWidget().hide()
        self.setWindowTitle("QtVTT")

        # XXX Forcing ini file for the time being for ease of debugging and
        #     tweaking, eventually move to native
        settings = QSettings('qtvtt.ini', QSettings.IniFormat)
        self.settings = settings
        
        # Fetch the MRUs from the app configuration and create the menu entry
        # for each of them
        for i in xrange(most_recently_used_max_count):
            filepath = settings.value("recentFilepath%d" % i)
            # Ignore empty or None filepaths
            if (not filepath):
                break
            self.recent_filepaths.append(filepath)
        # Set the first as most recent, will keep all entries the same but will
        # update all the menus
        if (len(self.recent_filepaths) > 0):
            self.setRecentFile(self.recent_filepaths[0])
            
        # Create all docks so geometry can be restored below
        # XXX There should be a better way of iterating through all keys?
        browser_keys = []
        for key in settings.allKeys():
            m = re.match("(docks/([^/]+))/class", key)
            if (m is None):
                continue
            key = m.group(1)
            className = settings.value("%s/class" % key)
            uuid = m.group(2)
            # XXX Remove once these are created dynamically
            if (className in ["ImageWidget", "QTreeWidget", "VTTGraphicsView"]):
                if (className == "ImageWidget"):
                    self.imageWidget.parent().setObjectName(uuid)
                elif (className == "QTreeWidget"):
                    self.tree.parent().setObjectName(uuid)
                elif (className == "VTTGraphicsView"):
                    self.graphicsView.parent().setObjectName(uuid)

                continue
            title = settings.value("%s/title" % key, "")

            # XXX Create some activate() method that takes care of the work at 
            #     newXXXX so this can be done generically
            if (className == "DocBrowser"):
                dock, view = self.createBrowser(uuid)
                browser_keys.append((view, key))

            elif (className == "EncounterBuilder"):
                dock, view = self.createEncounterBuilder(uuid)
            
            elif (className == "CombatTracker"):
                dock, view = self.createCombatTracker(uuid)

            else:
                assert False, "Unrecognized class %s" % className

        logger.info("Restoring window geometry and state")
        b = settings.value("layout/geometry")
        if (b):
            self.restoreGeometry(b)
        b = settings.value("layout/windowState")
        if (b):
            self.restoreState(b)
        
        for browser, key in browser_keys:
            data = settings.value("%s/state" % key)
            browser.restoreState(data)

        # Once all the elements have been created, set the scene which will
        # update all necessary windows (graphicsview, tree, imagewidget,
        # tracker, builder...)
        if (len(sys.argv) == 1):
            filepath = settings.value("recentFilepath0")
            if (filepath):
                logger.info("Restoring last campaign %r", filepath)
                self.loadScene(filepath)
                
            else:
                scene = load_test_scene()
                self.setScene(scene)
                
        else:
            self.loadScene(sys.argv[1])
            self.setRecentFile(sys.argv[1])

            
    def createMusicPlayer(self):
        self.player = QMediaPlayer()
        self.player.setVolume(50)
        self.playlist = QMediaPlaylist()
        # Set the playlist mode before attaching to the player, otherwise it's
        # ignored
        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
        self.player.setPlaylist(self.playlist)

        self.player.positionChanged.connect(self.musicPlayerPositionChanged)

    def musicPlayerPositionChanged(self, position):
        if (self.player.state() == QMediaPlayer.PlayingState):
            position = position / 10 ** 3
            duration = self.player.duration() / 10 ** 3
            
            nowStr = "%d:%02d" % (position / 60, position % 60)
            endStr = "%0d:%02d" % (duration / 60, duration % 60)
            self.statusMedia.setText(u"%s %s/%s" % (
                os.path.basename(self.playlist.currentMedia().canonicalUrl().path()),
                nowStr, endStr
            ))

        elif (self.player.state() == QMediaPlayer.PausedState):
            # Hook on paused vs. otherwise, because the code does a next after
            # pausing, which probably causes a stop with currentMedia already 
            # changed
            self.statusMedia.setText(u"%s paused" % (
                os.path.basename(self.playlist.currentMedia().canonicalUrl().path())
            ))

    def createActions(self):
        self.newAct = QAction("&New", self, shortcut="ctrl+n", triggered=self.newScene)
        self.openAct = QAction("&Open...", self, shortcut="ctrl+o", triggered=self.openScene)
        self.importDsAct = QAction("Import &Dungeon Scrawl...", self, triggered=self.importDs)
        self.saveAct = QAction("&Save", self, shortcut="ctrl+s", triggered=self.saveScene)
        self.saveAsAct = QAction("Save &As...", self, shortcut="ctrl+shift+s", triggered=self.saveSceneAs)
        self.exitAct = QAction("E&xit", self, shortcut="alt+f4", triggered=self.close)

        self.recentFileActs = []
        for i in range(most_recently_used_max_count):
            self.recentFileActs.append(
                    QAction(self, visible=False, triggered=self.openRecentFile))

        
        self.cutItemAct = QAction("Cut& Item", self, shortcut="ctrl+x", triggered=self.cutItem)
        self.copyItemAct = QAction("&Copy Item", self, shortcut="ctrl+c", triggered=self.copyItem)
        self.pasteItemAct = QAction("&Paste Item", self, shortcut="ctrl+v", triggered=self.pasteItem)
        self.deleteItemAct = QAction("&Delete Item", self, shortcut="del", triggered=self.deleteItem)
        self.importTokenAct = QAction("Import &Token...", self, shortcut="ctrl+t", triggered=self.importToken)
        self.importImageAct = QAction("Import &Image...", self, shortcut="ctrl+i", triggered=self.importImage)
        self.importMusicAct = QAction("Import &Music Track...", self, shortcut="ctrl+m", triggered=self.importMusic)
        self.clearWallsAct = QAction("Clear All &Walls...", self, triggered=self.clearWalls)
        self.clearDoorsAct = QAction("Clear All &Doors...", self, triggered=self.clearDoors)
        self.copyScreenshotAct = QAction("DM &View &Screenshot", self, shortcut="ctrl+alt+c", triggered=self.copyScreenshot)
        self.copyFullScreenshotAct = QAction("DM &Full Screenshot", self, shortcut="ctrl+shift+c", triggered=self.copyFullScreenshot)
        self.newCombatTrackerAct = QAction("New Combat Trac&ker", self, shortcut="ctrl+k", triggered=self.newCombatTracker)
        self.newEncounterBuilderAct = QAction("New Encounter Buil&der", self, shortcut="ctrl+d", triggered=self.newEncounterBuilder)
        self.newBrowserAct = QAction("New &Browser", self, shortcut="ctrl+b", triggered=self.newBrowser)
        self.lockFogCenterAct = QAction("&Lock Fog Center", self, shortcut="ctrl+l", triggered=self.lockFogCenter, checkable=True)
        self.nextTrackAct = QAction("N&ext Music Track", self, shortcut="ctrl+e", triggered=self.nextTrack)
        self.rewindTrackAct = QAction("Rewind Music Track", self, shortcut="ctrl+left", triggered=self.rewindTrack)
        self.forwardTrackAct = QAction("Forward Music Track", self, shortcut="ctrl+right", triggered=self.forwardTrack)
        self.closeWindowAct = QAction("&Close Window", self, triggered=self.closeWindow)
        self.closeWindowAct.setShortcuts(["ctrl+f4", "ctrl+w"])
        self.prevWindowAct = QAction("&Prev Window", self, shortcut="ctrl+shift+tab", triggered=self.prevWindow)
        self.nextWindowAct = QAction("&Next Window", self, shortcut="ctrl+tab", triggered=self.nextWindow)
        
        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
            triggered=QApplication.instance().aboutQt)


    def createMenus(self):
        fileMenu = QMenu("&File", self)
        fileMenu.setToolTipsVisible(True)
        fileMenu.addAction(self.newAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.importDsAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.saveAct)
        fileMenu.addAction(self.saveAsAct)
        fileMenu.addSeparator()
        for action in self.recentFileActs:
            fileMenu.addAction(action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAct)
        
        editMenu = QMenu("&Edit", self)
        editMenu.addAction(self.importImageAct)
        editMenu.addAction(self.importTokenAct)
        editMenu.addAction(self.importMusicAct)
        editMenu.addSeparator()
        editMenu.addAction(self.cutItemAct)
        editMenu.addAction(self.copyItemAct)
        editMenu.addAction(self.pasteItemAct)
        # XXX This interferes with del key on other windows (eg deleting rows in
        #     the encounter builder table by pressing "del"). Currently fixed by
        #     trapping the shortcut notification and declining it, find another
        #     way of fixing it? (note moving the act to the child widgets
        #     doesn't fix it? Add and remove actions per focused widget?)
        editMenu.addAction(self.deleteItemAct)
        editMenu.addSeparator()
        editMenu.addAction(self.clearDoorsAct)
        editMenu.addAction(self.clearWallsAct)
        editMenu.addSeparator()
        editMenu.addAction(self.copyScreenshotAct)
        editMenu.addAction(self.copyFullScreenshotAct)

        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.newBrowserAct)
        viewMenu.addAction(self.newCombatTrackerAct)
        viewMenu.addAction(self.newEncounterBuilderAct)
        viewMenu.addSeparator()
        viewMenu.addAction(self.lockFogCenterAct)
        viewMenu.addSeparator()
        viewMenu.addAction(self.nextTrackAct)
        viewMenu.addAction(self.rewindTrackAct)
        viewMenu.addAction(self.forwardTrackAct)
        viewMenu.addSeparator()
        viewMenu.addAction(self.closeWindowAct)
        viewMenu.addAction(self.prevWindowAct)
        viewMenu.addAction(self.nextWindowAct)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.aboutAct)
        helpMenu.addAction(self.aboutQtAct)

        bar = self.menuBar()
        bar.addMenu(fileMenu)
        bar.addMenu(editMenu)
        bar.addMenu(viewMenu)
        bar.addMenu(helpMenu)
        

    def createStatus(self):
        frame_style = QFrame.WinPanel | QFrame.Sunken

        # Can't set sunken style on QStatusBar.showMessage, use a widget and
        # reimplement showMessage and clearMessage
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(self.clearMessage)
        self.status_message_timer = timer

        self.status_message_widget = QLabel()
        self.status_message_widget.setFrameStyle(frame_style)
        self.statusBar().addWidget(self.status_message_widget, 1)

        self.statusMedia = QLabel()
        self.statusMedia.setFrameStyle(frame_style)
        self.statusBar().addPermanentWidget(self.statusMedia)

        self.statusScene = QLabel()
        self.statusScene.setFrameStyle(frame_style)
        self.statusBar().addPermanentWidget(self.statusScene)


    def about(self):
        QMessageBox.about(self, "About QtVTT",
                "<p>Simple no-frills <b>virtual table top</b>:"
                "<ul>"
                "<li> token collision "
                "<li> line of sight"
                "<li> remote viewing via http"
                "<li> <a href=\"https://app.dungeonscrawl.com/\">Dungeon Scrawl</a> import"
                "<li> combat tracker"
                "<li> encounter builder"
                "<li> and more"
                "</ul>"
                "Visit the <a href=\"https://github.com/antoniotejada/QtVTT\">github repo</a> for more information</p>")
                

    def closeEvent(self, event):
        logger.info("closeEvent")
        
        # XXX May want to have a default and then a per-project setting 
        # XXX May want to have per resolution settings
        settings = self.settings

        # XXX Should also save and restore the zoom positions, scrollbars, tree
        # folding state

        logger.info("Storing window geometry and state")
        settings.setValue("layout/geometry", self.saveGeometry())
        settings.setValue("layout/windowState", self.saveState())
        
        logger.info("Storing most recently used")
        for i, filepath in enumerate(self.recent_filepaths):
            settings.setValue("recentFilepath%d" %i, filepath)

        logger.info("Deleting dock entries")
        for dockKey in [key for key in settings.allKeys() if key.startswith("docks/")]:
            logger.info("Deleting dock entry %s", key)
            settings.remove(dockKey)

        logger.info("Storing dock widgets")
        for dock in self.findChildren(QDockWidget):
            logger.info("Found dock %s %s", dock, dock.widget())
            uuid = dock.objectName()
            key = "docks/%s" % uuid
            settings.setValue("%s/class" % key, dock.widget().__class__.__name__)
            # XXX Actually implement saveState everywhere
            if (getattr(dock.widget(), "saveState", None) is not None):
                state = dock.widget().saveState()
                settings.setValue("%s/state" % key, state)
    
        super(VTTMainWindow, self).closeEvent(event)
        
    def showMessage(self, msg, timeout_ms=2000):
        logger.info("%s %d", msg, timeout_ms)
        self.status_message_timer.stop()
        self.status_message_widget.setText(msg)
        if (timeout_ms > 0):
            self.status_message_timer.start(timeout_ms)
            
    def clearMessage(self):
        logger.info("")
        self.status_message_widget.setText("")

    def setScene(self, scene, filepath = None):
        logger.debug("")

        print_gc_stats()

        self.scene = scene

        # XXX Since there are so many setScene spurious calls, keep the focus if
        #     locked across setScene calls. Specifically CombatTracker calls
        #     setScene on every update of combat stats. Remove once updates are
        #     piecemeal
        fogCenter = None
        if ((self.gscene is not None) and self.gscene.getFogCenterLocked()):
            fogCenter = self.gscene.getFogCenter()

        # XXX Verify if all this old gscene cleaning up is necessary and/or
        #     enough
        if (self.gscene is not None):
            self.gscene.clear()
            self.gscene.setParent(None)
        self.fog_polys = []

        gscene = VTTGraphicsScene(scene)
        if (fogCenter is not None):
            gscene.setFogCenterLocked(True)
            gscene.setFogCenter(fogCenter)
        gscene.changed.connect(self.sceneChanged)
        # XXX It's not clear the BSP is helping on dynamic scenes with fog
        #     (although the fog is not selectable so it shouldn't be put in the
        #     bsp?)
        ##gscene.setItemIndexMethod(QGraphicsScene.NoIndex)

        self.campaign_filepath = filepath
        if (filepath is not None):
            self.setWindowTitle("QtVTT - %s" % os.path.basename(filepath))

        else:
            self.setWindowTitle("QtVTT")

        # Updating for every single element of the scene is unnecessary, block
        # signals, unblock when done and update manually below
        gscene.blockSignals(True)
        self.populateGraphicsScene(gscene, scene)
        self.graphicsView.setScene(gscene)
        gscene.blockSignals(False)

        self.player.stop()
        self.playlist.clear()
        for track in scene.music:
            logger.info("Adding music track %r", track.filepath)
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(track.filepath)))
        

        self.gscene = gscene

        # XXX These should go through signals
        self.updateTree()
        self.updateCombatTrackers()
        self.updateImage()

        # Repaint the image widget and start with some sane scroll defaults
        
        # XXX Note this will reset the scroll unexpectedly until several scene
        #     modification functions stop calling setScene (clearing all
        #     walls/doors, deleting item, etc)
        self.imageWidget.setFitToWindow(True)

        print_gc_stats()

    def openRecentFile(self):
        logger.info("openRecentFile")
        action = self.sender()
        if (action is not None):
            self.loadScene(action.data())
            self.setRecentFile(action.data())

    def setRecentFile(self, filepath):
        """
        See
        https://github.com/baoboa/pyqt5/blob/master/examples/mainwindows/recentfiles.py

        Note adding actions dynamically didn't work because when the action is
        removed from the menu, the action seems to remain referenced somewhere
        in the system, which causes all the actions to refer to the first recent
        file ever registered.

        XXX The above problem could be related to using lambdas in a loop?
        """
        logger.info("setRecentFile %r", filepath)

        abspath = os_path_abspath(filepath)

        # The list doesn't contain duplicates, if this is a duplicate remove
        # this one from the list first and then insert at the top
        i = index_of(self.recent_filepaths, abspath)
        if (i != -1):
            self.recent_filepaths.pop(i)
        self.recent_filepaths.insert(0, abspath)
        
        if (len(self.recent_filepaths) > most_recently_used_max_count):
            self.recent_filepaths.pop(-1)

        # Update all the actions
        for i in xrange(most_recently_used_max_count):
            if (i < len(self.recent_filepaths)):
                filepath = self.recent_filepaths[i]
                logger.info("Setting MRU %s", filepath)
                
                self.recentFileActs[i].setText(os.path.basename(filepath))
                self.recentFileActs[i].setToolTip(filepath)
                self.recentFileActs[i].setData(filepath)
                self.recentFileActs[i].setVisible(True)

            else:
                self.recentFileActs[i].setVisible(False)

    def newScene(self):
        scene = Struct()
        
        # XXX Create from an empty json so there's a single scene init path?
        #     Also, the "empty" json could have stock stuff like tokens, etc
        scene.map_doors = []
        scene.map_walls = []
        scene.cell_diameter = 70
        scene.map_tokens = []
        scene.map_images = []
        scene.music = []

        self.setScene(scene)

    
    def importToken(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        # XXX Get supported extensions from Qt
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Token", dirpath, "Images (*.png *.jpg *.jpeg *.jfif *.webp)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return
        
        # XXX This code is duplicated at load time
        s = Struct(**{
            # Center the token in the current viewport
            "scene_pos": 
                qtuple(self.graphicsView.mapToScene(qSizeToPoint(self.graphicsView.size()/2.0)))
            , 
            # XXX Fix all float casting this malarkey, cast it in whatever
            #     operation needs it, not in the input data
            "scale": float(self.scene.cell_diameter), 
            "name": os.path.splitext(os.path.basename(filepath))[0], 
            # XXX Fix all the path mess for embedded assets
            "filepath": os.path.relpath(filepath),
            "hidden" : False,
            "ruleset_info" : Struct(**default_ruleset_info)
        })
        self.scene.map_tokens.append(s)
        # XXX Use something less heavy handed than setScene
        self.setScene(self.scene, self.campaign_filepath)

    def importImage(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        # XXX Get supported extensions from Qt
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Image", dirpath, "Images (*.png *.jpg *.jpeg *.jfif *.webp)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        # Try to guess the number of cells
        m = re.match(r".*\D+(\d+)x(\d+)\D+", filepath)
        if (m is not None):
            img_size_in_cells = (int(m.group(1)), int(m.group(2)))
            logger.debug("img size in cells %s", img_size_in_cells)

        else:
            size = QImage(filepath).size()
            text, ok = QInputDialog.getText(
                self,
                "Image size in cells (%dx%d)" % (size.width(), size.height()), 
                "Cells (width, height):", QLineEdit.Normal, 
                "%d, %d" % (round(float(size.width()) / self.scene.cell_diameter), 
                    round(float(size.height()) / self.scene.cell_diameter))
            )
            if ((not ok) or (text == "")):
                return
            img_size_in_cells = [int(i) for i in text.split(",")]

        s = Struct(**{
            # XXX Fix all the path mess for embedded assets
            "filepath" :  os.path.relpath(filepath), 
            "scene_pos" : [0.0,0.0],
            # Note this stores a resolution independent scaling, it has
            # to be divided by the width at load time
            # XXX This assumes the scaling preserves the aspect ratio, may 
            #     need to store scalex and scaly
            # Fit it on the viewport by default
            "scale" : float(self.scene.cell_diameter * img_size_in_cells[0])
        })
        self.scene.map_images.append(s)

        # XXX Use something less heavy handed
        self.setScene(self.scene, self.campaign_filepath)

    def importMusic(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        # XXX Get supported extensions from Qt
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Music Track", dirpath, "Music Files (*.mp3 *.ogg)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        # XXX Allow music area, etc
        # XXX Get metadata from music
        s = Struct(**{
            # XXX Fix all the path mess for embedded assets
            "filepath" :  os.path.relpath(filepath), 
            "name" : os.path.splitext(os.path.basename(filepath))[0],
        })
        self.scene.music.append(s)

        # XXX Use something less heavy handed
        self.setScene(self.scene, self.campaign_filepath)
        
    def copyItem(self, cut=False):
        gscene = self.gscene
        if (self.graphicsView.hasFocus() and (len(gscene.selectedItems()) > 0)):
            logger.info("Copying %d tokens", len(gscene.selectedItems()))
            map_tokens = []
            for item in gscene.selectedItems():
                # XXX Copy JSON MIME?
                map_token = item.data(0)
                map_tokens.append(map_token)
            js = json.dumps({ "tokens" :  map_tokens, "cell_diameter" : self.scene.cell_diameter }, indent=2, cls=JSONStructEncoder)
            logger.info("Copied to clipboard %s", js)
            qApp.clipboard().setText(js)

    def pasteItem(self):
        # XXX Use mime and check the mime type instead of scanning the text
        if (self.graphicsView.hasFocus() and ('"tokens":' in qApp.clipboard().text())):
            js = json.loads(qApp.clipboard().text())
            map_tokens = js["tokens"]
            cell_diameter = float(js["cell_diameter"])
            
            logger.info("Pasting %d tokens", len(map_tokens))
            # XXX This should convert any dicts to struct
            # XXX Needs to reroll HP? (not clear it's desirable)
            # Unselect all items, will leave pasted items as selected so they
            # can be easily moved/deleted
            self.gscene.clearSelection()
            for map_token in map_tokens:
                logger.debug("Pasting token %s", map_token)
                ruleset_info = map_token["ruleset_info"]
                map_token = Struct(**map_token)
                map_token.ruleset_info = Struct(**ruleset_info)
                # Tokens are normally sized wrt cell diameters, this scene may
                # have a different cell diameter from the scene the token was 
                # copied from, scale from incoming to current cell diameter
                map_token.scale = map_token.scale * self.scene.cell_diameter/ cell_diameter
                # XXX This should reset the position too?
                self.scene.map_tokens.append(map_token)
                token = self.gscene.addToken(map_token)
                # Select the pasted item
                token.setSelected(True)

    
    def cutItem(self):
        self.copyItem()
        self.deleteItem()

    def deleteItem(self):
        changed = False
        if (self.tree.hasFocus()):
            item = self.tree.currentItem()
            # XXX Check the parent in case this is in some subitem?
            
            # XXX This could delete whole sections if the returned parent is a
            #     struct, would need to put an empty one instead (and probably
            #     ask for confirmation)
            data = item.data(0, Qt.UserRole)
            l = find_parent(self.scene, data)
            if ((l is not None) and isinstance(l, (list, tuple))):
                logger.info("Deleting item %s from list %s", item, l)
                l.remove(data)
                changed = True
                    
        elif (self.graphicsView.hasFocus() and (len(self.gscene.selectedItems()) > 0)):
            # XXX This only expects tokens for the time being
            for item in self.gscene.selectedItems():
                logger.info("Deleting graphicsitem %s", item.data(0))
                self.scene.map_tokens.remove(item.data(0))
            changed = True
            
        if (changed):
            # XXX Use something less heavy-handed
            self.setScene(self.scene, self.campaign_filepath)

    def clearWalls(self):
        if (QMessageBox.question(self, "Clear all walls", 
            "Are you sure you want to clear all walls in the scene?") != QMessageBox.Yes):
            return

        self.scene.map_walls = []

        # XXX Use something less heavy handed
        self.setScene(self.scene, self.campaign_filepath)

    def clearDoors(self):
        if (QMessageBox.question(self, "Clear all doors", 
            "Are you sure you want to clear all doors in the scene?") != QMessageBox.Yes):
            return
        self.scene.map_doors = []

        # XXX Use something less heavy handed
        self.setScene(self.scene, self.campaign_filepath)

    def copyScreenshot(self):
        logger.info("copyScreenshot")
        qpix = self.graphicsView.viewport().grab()
        
        qApp.clipboard().setPixmap(qpix)

    def copyFullScreenshot(self):
        logger.info("copyFullScreenshot")
        gscene = self.gscene
        img_scale = 1.0
        qim = QImage(gscene.sceneRect().size().toSize().expandedTo(QSize(1, 1)) * img_scale, QImage.Format_ARGB32)
        p = QPainter(qim)
        gscene.render(p)
        p.end()
        
        qApp.clipboard().setImage(qim)

    def wrapInDockWidget(self, view, title, uuid = None):
        def topLevelChanged(floating):
            logger.info("floating %s", floating)
            dw = self.sender()
            # XXX In theory this puts maximize buttons on floating docks, but
            #     the window disappears, investigate
            #     For the time being the imageWidget dock is 
            # See https://stackoverflow.com/questions/50531257/maximize-and-minimize-buttons-in-an-undocked-qdockwidget
            #if (floating):
            # if (dw.isFloating()):
            #     dw.setWindowFlags(Qt.CustomizeWindowHint |
            #              Qt.Window | Qt.WindowMinimizeButtonHint |
            #              Qt.WindowMaximizeButtonHint |
            #              Qt.WindowCloseButtonHint
            #     )
            #    #dw.setWindowFlags(dw.windowFlags() | Qt.WindowMaximizeButtonHint)

        logger.info("%s %s", view, title)
        dock = QDockWidget(title, self)
        # dock.topLevelChanged.connect(topLevelChanged)
        # Set the object name, it's necessary so Qt can save and restore the
        # state in settings, and use a UID so window positions are restored
        # store the UUID in the config file
        if (uuid is None):
            uuid = QUuid.createUuid().toString()
        dock.setObjectName(uuid)
        dock.setWidget(view)
        dock.setFloating(False)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setAttribute(Qt.WA_DeleteOnClose)

        self.addDockWidget(Qt.TopDockWidgetArea, dock)

        return dock

    def findBrowser(self):
        logger.info("")
        # Use object name as it's more deterministic than windowTitle which may
        # have collisions that cause flip flopping across invocations
        browsers = sorted([browser for browser in self.findChildren(DocBrowser)], 
            cmp=lambda a,b: cmp(self.findParentDock(a).objectName(), self.findParentDock(b).objectName()))
        # If there are no browsers or ctrl+shift is pressed, create a new one
        ctrlShift = int(Qt.ControlModifier | Qt.ShiftModifier)
        if ((len(browsers) == 0) or 
            # XXX Have the signal send new/existing browser?
            (int(qApp.keyboardModifiers()) & ctrlShift) == ctrlShift):
            dock, browser = self.createBrowser()

        else:
            for browser in browsers:
                # Note isVisible returns true for all tabified docks, use
                # visibleRegion empty checks instead
                if (not browser.visibleRegion().isEmpty()):
                    # If a browser is visible, use that one
                    break

            else:
                # If a browser was found but it's not visible, bring to front
                browser = browsers[0]
                dock = self.findParentDock(browser)
                # Raise the tab, keep the same focus
                dock.show()
                dock.raise_()

            # Ctrl is pressed and a browser was found, create a new tab
            if ((int(qApp.keyboardModifiers()) & Qt.ControlModifier) != 0):
                dock, newBrowser = self.createBrowser()
                self.tabifyDockWidget(self.findParentDock(browser), dock)
                # Raise the tab, keep the same focus
                dock.show()
                dock.raise_()
                browser = newBrowser

        return browser

    def browseMonster(self, filepath_or_id):
        logger.info("%s" % filepath_or_id)
        browser = self.findBrowser()
        filepath = filepath_or_id
        if (not filepath_or_id.lower().endswith((".htm", ".html"))):
            logger.info("Searching for monster with id %s", filepath_or_id)
            # XXX Cache this somewhere
            filepath = os.path.join("_out", "monsters2.csv")
            with open(filepath, "rb") as f:
                rows = list(csv.reader(f, delimiter="\t"))
                headers = rows[0]
                linkIndex = index_of(headers, "Link")
                nameIndex = index_of(headers, "Name")
                for row in rows:
                    if (filepath_or_id == row[nameIndex]):
                        logger.info("Found monster id %s filepath %s", filepath_or_id, row[linkIndex])
                        filepath = row[linkIndex]
                        break
                else:
                    filepath = "Monsters1/MM00000.htm"
        
        browser.setSourcePath(os.path.join(DocBrowser.docDirpath, filepath))

    def browseQuery(self, query):
        self.info("query %s", query)
        browser = self.findBrowser()
        browser.lineEdit.setText(query)

    def createCombatTracker(self, uuid = None):
        tracker = CombatTracker()
        dock = self.wrapInDockWidget(tracker, "Combat Tracker", uuid)
        
        tracker.sceneChanged.connect(self.updateFromCombatTracker)

        tracker.browseMonster.connect(self.browseMonster)

        return dock, tracker

    def newCombatTracker(self):
        # XXX Arguably only one combattracker should exist per scene or just
        #     one, for the active scene, since they all have the same info and
        #     they don't update each other?
        #     Unless the group is split in two rooms? But currently all 
        #     the tokens in the current map appear in the combat tracker, should
        #     have room support? Currently this can be done with two trackers and
        #     disabling the tokens not in whatever room and do the opposite on 
        #     the other tracker
        logger.info("")
        dock, tracker = self.createCombatTracker()
        
        return dock, tracker

    def createEncounterBuilder(self, uuid = None):
        builder = EncounterBuilder()
        dock = self.wrapInDockWidget(builder, "Encounter Builder", uuid)

        builder.addTokensButton.clicked.connect(lambda : self.addTokensFromEncounter(builder))
        builder.browseMonster.connect(self.browseMonster)
        
        return dock, builder

    def newEncounterBuilder(self):
        logger.info("")
        dock, builder = self.createEncounterBuilder()
        
        return dock, builder

    def createBrowser(self, uuid = None):
        browser = DocBrowser()
        dock = self.wrapInDockWidget(browser, "Browser", uuid)
        
        def sourceChanged(url):
            dock.setWindowTitle("%s" % browser.getSourceTitle())
            
        browser.textEdit.sourceChanged.connect(sourceChanged)
        
        return dock, browser

    def newBrowser(self, uuid = None):
        logger.info("")
        dock, browser = self.createBrowser()

        return dock, browser

    def lockFogCenter(self):
        self.gscene.setFogCenterLocked(not self.gscene.getFogCenterLocked())
        self.lockFogCenterAct.setChecked(self.gscene.getFogCenterLocked())
        self.gscene.makeDirty()
        self.gscene.invalidate()

    def nextTrack(self):
        # XXX Hook status changed to the statusbar
        if (self.player.state() != QMediaPlayer.PlayingState):
            self.player.play()
            mediaName = self.playlist.currentMedia().canonicalUrl()
            self.showMessage("Playing %r" % mediaName)
            logger.info("Playing %r", mediaName)
            
        else:
            mediaName = self.playlist.currentMedia().canonicalUrl()
            self.player.pause()
            self.playlist.next()
            self.showMessage("Stopping %r" % mediaName)

    def forwardTrack(self):
        if (self.player.state() != QMediaPlayer.PlayingState):
            self.player.play()
        mediaName = self.playlist.currentMedia().canonicalUrl()
        logger.info("%r", mediaName)

        delta = 5 * 10**3
        newPosition = self.player.position() + delta
        self.showMessage("Forwarding %r %d/%d" % (mediaName, newPosition / 10**3, self.player.duration()/10**3))
        
        self.player.setPosition(newPosition)

    def rewindTrack(self):
        if (self.player.state() != QMediaPlayer.PlayingState):
            self.player.play()
        mediaName = self.playlist.currentMedia().canonicalUrl()
        logger.info("%r", mediaName)
        delta = -5 * 10**3
        newPosition = self.player.position() + delta
        self.showMessage("Rewinding %r %d/%d" % (mediaName, newPosition / 10**3, self.player.duration()/10**3))
        
        self.player.setPosition(newPosition)

    def focusChanged(self, old, new):
        logger.info("")
        if (old is not None):
            dock = self.findParentDock(old)
            if (dock is not None):
                self.setDockStyle(dock, False)

        if (new is not None):
            dock = self.findParentDock(new)
            if (dock is not None):
                self.setDockStyle(dock, True)

    def setDockStyle(self, dock, focused):
        logger.info("%s, %s", dock.windowTitle(), focused)
        # XXX Setting this stylesheet causes random font resizing on table
        #     cellwidgets when clicked on the table row? (no longer noticeable
        #     since the table doesn't use QLabels for hyperlinks) Do this with
        #     palette changes?
        if (focused):
            # See palette CSS reference at https://doc.qt.io/qt-5/stylesheet-reference.html
            # There's no way of telling focused and non focused dock widgets
            # apart, different style sheets need to be set as they are focused
            # in or out
            dock.setStyleSheet("""
                QDockWidget::title { text-align: left; /* align the text to the
                    left */ background: palette(highlight); padding-left: 5px;
                }
                /* Setting the title text color has no effect, set the global
                color instead */ QDockWidget { color: palette(highlighted-text);
                }
            """)

            # None of active or focus can tell the difference between tabs in
            # focus and non-focus docks, search the QTabBar associated to this
            # dock.
            tabBar = qFindTabBarFromDockWidget(dock)
            if (tabBar is not None):
                tabBar.setStyleSheet("""
                    QTabBar::tab:selected {
                        background: palette(highlight);
                        color: palette(highlighted-text); 
                    }
                """)
            
        else:
            # Restore the style sheet
            dock.setStyleSheet("""
                QDockWidget::title {
                    text-align: left;  /* align the text to the left */
                    padding-left: 5px;
                }
            """)
            tabBar = qFindTabBarFromDockWidget(dock)
            if (tabBar is not None):
                tabBar.setStyleSheet("""
                    QTabBar::tab:selected {
                    }
                """)

    def focusDock(self, dock):
        logger.info("Focusing %s", dock.windowTitle())
        
        # XXX Missing focusing on tab click, see https://stackoverflow.com/questions/51215635/notification-when-qdockwidgets-tab-is-clicked
        #     It's not clear if the above can be used when there are multiple
        #     dockwidgets, the tabbar doesn't seem to have any pointers to the
        #     dock widget?
        # XXX Clicking on empty encounter table swallows ctrl+tab navigation, investigate
        # XXX Missing ctrl+tab navigation when central widget is focused, investigate
        dock.show()
        dock.raise_()
        dock.widget().setFocus(Qt.TabFocusReason)

    def findParentDock(self, widget):
        logger.info("")
        
        # XXX The parent of a dock widget is always? the dock already, this is
        #     probably overkill
        parent = widget
        while ((parent is not None) and (not isinstance(parent, QDockWidget))):
            # XXX When playing with table cell widgets and focusing back on the
            #     table, the focus is not on a window, investigate?
            logger.info("dock stack %r", getattr(parent, "windowTitle", "Not a window"))
            parent = parent.parent()

        return parent

    def findFocusedDock(self):
        return self.findParentDock(self.focusWidget())
        
    def getDocksInTabOrder(self):
        """
        Rerturn the docks in ctrl+tab/ctrl+shift+tab order. This is used for
        tabbing to the next/prev dock, so it must return a consistent order
        between invocations.
        """
        logger.info("")

        # Findchildren is not consistent across calls, sort by object name
        # (better than window title since there can be multiple windows with the
        # same title, which would make the tab order to flip flop)
        allDocks = sorted([dock for dock in self.findChildren(QDockWidget)], cmp=lambda a,b: cmp(a.objectName(), b.objectName()))

        visited = set()
        docks = []
        # Note findChildren order is not constant across invocations
        # Return docks grouped by tabs, ordered alphabetically
        for dock in allDocks:
            if (dock in visited):
                continue
            
            # Visit tabs, these have consistent left to right ordering, no need
            # to sort them
            # The list doesn't include the dock passed as parameter and it's
            # empty if the dock is not tabified
            # Because the list doesn't include the passed dock, its position in
            # the list is unknown, get the list for another dock and compare
            # the first list and the second
            # XXX There should be a better way of doing this? there's a tabbar
            #     child of the mainwindow, but it's not clear how to link the
            #     tabbar to the given dock (by tab title? by position?)
            tabbedDocks = self.tabifiedDockWidgets(dock)
            if (len(tabbedDocks) <= 1):
                # Order doesn't matter
                tabbedDocks.append(dock)
            
            else:
                # Get the list again and find out where the original dock falls
                dock2 = tabbedDocks[0]
                tabbedDocks2 = self.tabifiedDockWidgets(dock2)
                if (tabbedDocks2[0] is dock):
                    # dock is the first in the list, so it's not known if dock
                    # was before or after dock2, need another check
                    dock3 = tabbedDocks[1]
                    tabbedDocks3 = self.tabifiedDockWidgets(dock3)
                    tabbedDocks = tabbedDocks3[0:2] + tabbedDocks2[1:]
                    
                else:
                    tabbedDocks = [dock2] + tabbedDocks2

            
            for dock in tabbedDocks:
                visited.add(dock)
                docks.append(dock)

        return docks

    def closeWindow(self):
        logger.info("current widget %s", self.focusWidget().windowTitle())
        focusedDock = self.findFocusedDock()

        # If on a tabbed dock with prev dock outside the tabbed dock, and the next tab
        # is in the tabbed dock, go to the next tab, otherwise go to the prev
        # tab
        tabbedDocks = self.tabifiedDockWidgets(focusedDock)
        if (len(tabbedDocks) > 0):
            
            docks = self.getDocksInTabOrder()
            i = index_of(docks, focusedDock)
            if ((docks[i-1] not in tabbedDocks) and (docks[i+1] in tabbedDocks)):
                logger.info("Closing tabbed dock %s, going to next %s", focusedDock.windowTitle(), docks[i+1])
                self.nextWindow()

            else:
                logger.info("Closing tabbed dock %s, going to prev %s", focusedDock.windowTitle(), docks[i-1])
                self.prevWindow()
        else:
                
            self.prevWindow()

        focusedDock.close()

    def prevWindow(self):
        # XXX Ideally this should prev to the prev tab and only to the prev
        #     window once done with tabs?
        logger.info("current widget %s", self.focusWidget().windowTitle())
        focusedDock = self.findFocusedDock()

        if (focusedDock is not None):
            logger.info("Parent dock %r", focusedDock.windowTitle())

            prev = None
            docks = self.getDocksInTabOrder()
            for dock in docks:
                logger.info("Found dock %s focus %s widget %s" % (dock.windowTitle(), dock.hasFocus(), dock.widget().hasFocus()))
                if (dock is focusedDock):
                    if (prev is None):
                        prev = docks[-1]
                    self.focusDock(prev)
                    
                    break
                prev = dock

    def nextWindow(self):
        # XXX Ideally this should next to the next tab and only to the next
        #     window once done with tabs?
        logger.info("current %s", self.focusWidget().windowTitle())

        focusedDock = self.findFocusedDock()

        if (focusedDock is not None):
            logger.info("Focused dock %r", focusedDock.windowTitle())

            prev = None
            docks = self.getDocksInTabOrder()
            for dock in docks:
                logger.info("Found dock %s focus %s widget %s" % (dock.windowTitle(), dock.hasFocus(), dock.widget().hasFocus()))
                if ((prev is focusedDock) or (dock is docks[-1])):
                    if (prev is not focusedDock):
                        dock = docks[0]
                    self.focusDock(dock)
                        
                    break
                prev = dock
            
        
    def importDs(self):
        # Get the ds filename
        dirpath = os.path.curdir if self.campaign_filepath is None else self.campaign_filepath
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Dungeon Scrawl data file", dirpath, "Dungeon Scrawl (*.ds)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        ds_filepath = filepath

        # Try to load a similarly name and dated png, otherwise ask for the png
        # filename
        l = os.listdir(os.path.dirname(ds_filepath))
        map_filepath = None
        prefix = os.path.splitext(os.path.basename(ds_filepath))[0]
        for filename in l:
            if (filename.startswith(prefix) and filename.endswith(".png")):
                this_filepath = os.path.join(os.path.dirname(ds_filepath), filename)
                logger.info("Found png candidate %r", this_filepath)
                if ((map_filepath is None) or 
                    # There can be multiple revisions of the png file with
                    # different cell sizes or (n) suffix from downloading with
                    # the browser, pick the one that has the date closest to the
                    # ds file
                    (
                     abs(os.path.getmtime(this_filepath) - os.path.getmtime(ds_filepath)) < 
                     abs(os.path.getmtime(map_filepath) - os.path.getmtime(ds_filepath))
                    )
                ):
                    logger.info("Keeping better matched %r than %r", map_filepath, map_filepath)
                    map_filepath = this_filepath

        if (map_filepath is None):
            dirpath = os.path.curdir if self.campaign_filepath is None else self.campaign_filepath
            filepath, _ = QFileDialog.getOpenFileName(self, "Import Dungeon Scrawl battlemap", dirpath, "PNG file (*.png)")

            if (filepath == ""):
                filepath = None

            map_filepath = filepath

        # Copy the png to the output dir for the time being so the qvt file
        # is not left with a file pointing eg to some download directory
        # XXX This won't be necessary once assets are properly embedded in the qvt
        if (map_filepath is not None):
            copy_filepath = os.path.join("_out", os.path.basename(map_filepath))
            if (os.path.normpath(os_path_abspath(map_filepath)) != os.path.normpath(os_path_abspath(copy_filepath))):
                os_copy(map_filepath, copy_filepath)
                map_filepath = copy_filepath

        # Note the map filepath can be none if they decided to load only walls 
        scene = load_ds(ds_filepath, map_filepath)
        # XXX Setting a filepath here skips the most recently used and won't ask
        #     for confirmation when saving, needs better flow
        # XXX Create this in the out directory for the time being
        filepath = os.path.splitext(os.path.join("_out", os.path.basename(ds_filepath)))[0] + ".qvt"
        self.setScene(scene, filepath)
        
        
    def loadScene(self, filepath):
        self.showMessage("Loading %r" % filepath)

        scene = Struct()

        with zipfile.ZipFile(filepath, "r") as f:
            # Read the map and the assets
            with f.open("campaign.json", "r") as ff:
                js = json.load(ff)
        
        # XXX Right now this expects only one scene
        js = js["scenes"][0]
        
        # XXX Have a generic json to struct conversion
        # Convert door lines to door objects
        # XXX Remove once all maps have door objects
        if ((len(js["map_doors"]) > 0) and (isinstance(js["map_doors"][0], list))):
            scene.map_doors = [Struct(lines=door, open=False) for door in js["map_doors"]]
        
        else:
            scene.map_doors = [Struct(**map_door) for map_door in js["map_doors"]]
        scene.map_walls = js["map_walls"]
        scene.cell_diameter = js["cell_diameter"]
        scene.map_tokens = [Struct(**map_token) for map_token in js["map_tokens"]]
        # XXX This should convert any dicts to struct
        for map_token in scene.map_tokens:
            if (getattr(map_token, "ruleset_info", None) is None):
                map_token.ruleset_info = Struct(**default_ruleset_info)
            else:
                # Merge with the default in case new fields were added
                d = dict(default_ruleset_info)
                d.update(map_token.ruleset_info)
                map_token.ruleset_info = Struct(**d)
        # XXX Remove once all files have this data
        if ((len(scene.map_tokens) > 0) and (not hasattr(scene.map_tokens[0], "hidden"))):
            for map_token in scene.map_tokens:
                map_token.hidden = False
        scene.map_images = [Struct(**map_image) for map_image in js["map_images"]]
        scene.music = [Struct(**track) for track in js.get("music", [])]

        self.setScene(scene, filepath)

        self.showMessage("Loaded %r" % filepath)


    def openScene(self):

        dirpath = os.path.curdir if self.campaign_filepath is None else self.campaign_filepath
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", dirpath, "Campaign (*.qvt)" )

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        self.setRecentFile(filepath)

        self.loadScene(filepath)

    def updateCombatTrackers(self):
        # XXX Remove this loop once trackers register to the signal
        for tracker in self.findChildren(CombatTracker):
            tracker.setScene(self.scene)

    def updateFromCombatTracker(self):
        logger.debug("")
        # XXX setScene at this point causes the focus to be lost on the DM view,
        #     which is undesirable because eg gets lost on every stat
        #     modification (attack roll, hitpoint update, etc), and locking
        #     the focus doesnt' survive across setScene either.
        self.updateTree()
        self.updateImage()
        # XXX Use something less heavy handed than setScene
        self.setScene(self.scene, self.campaign_filepath)

    def updateTree(self):
        # XXX Have tabs to tree by folder, by asset type, by layer (note they can 
        #     be just more dockwidgets, since dockwidgets can be tabbed)

        # XXX This should probably be a treeview that fetches directly from the
        #     model
        tree = self.tree

        tree.clear()
        tree.setColumnCount(1)
        tree.setHeaderHidden(True)

        scene_item = QTreeWidgetItem(["Scene 1"])
        tree.addTopLevelItem(scene_item)

        scene_item.addChild(QTreeWidgetItem(["%d" % self.scene.cell_diameter]))

        music = getattr(self.scene, "music", [])
        folder_item = QTreeWidgetItem(["Music (%d)" % len(music)])
        scene_item.addChild(folder_item)
        for track in music:
            subfolder_item = QTreeWidgetItem(["%s" % track.name])
            subfolder_item.setData(0, Qt.UserRole, track)

            item = QTreeWidgetItem(["%s" % track.filepath])
            subfolder_item.addChild(item)
            
            folder_item.addChild(subfolder_item)
        
        folder_item = QTreeWidgetItem(["Walls (%d)" % len(self.scene.map_walls)])
        scene_item.addChild(folder_item)
        for wall in self.scene.map_walls: 
            child = QTreeWidgetItem(["%s" % (wall,)])
            child.setData(0, Qt.UserRole, wall)

            folder_item.addChild(child)
        
        folder_item = QTreeWidgetItem(["Doors (%d)" % len(self.scene.map_doors)])
        scene_item.addChild(folder_item)
        for door in self.scene.map_doors: 
            child = QTreeWidgetItem(["%s%s" % ("*" if door.open else "", door.lines)])
            child.setData(0, Qt.UserRole, door)

            folder_item.addChild(child)

        folder_item = QTreeWidgetItem(["Images (%d)" % len(self.scene.map_images)])
        scene_item.addChild(folder_item)
        for image in self.scene.map_images:
            subfolder_item = QTreeWidgetItem(["%s" % os.path.basename(image.filepath)])
            subfolder_item.setData(0, Qt.UserRole, image)

            item = QTreeWidgetItem(["%s" % (image.scene_pos,)])
            subfolder_item.addChild(item)
            item = QTreeWidgetItem(["%s" % image.scale])
            subfolder_item.addChild(item)

            folder_item.addChild(subfolder_item)

        folder_item = QTreeWidgetItem(["Tokens (%d)" % len(self.scene.map_tokens)])
        scene_item.addChild(folder_item)
        for token in self.scene.map_tokens:
            # XXX Use () for hidden and * for current token?
            subfolder_item = QTreeWidgetItem(["%s%s" % ("*" if token.hidden else "", token.name, )])
            subfolder_item.setData(0, Qt.UserRole, token)

            item = QTreeWidgetItem([os.path.basename(token.filepath)])
            subfolder_item.addChild(item)
            item = QTreeWidgetItem(["%s" % (token.scene_pos,)])
            subfolder_item.addChild(item)
            item = QTreeWidgetItem(["%s" % token.scale])
            subfolder_item.addChild(item)

            folder_item.addChild(subfolder_item)

    def saveScene(self):
        filepath = self.campaign_filepath
        if (filepath is None):
            #dirpath = os.path.curdir if filepath is None else filepath
            filepath = os.path.join("_out", "campaign.qvt")
            filepath, _ = QFileDialog.getSaveFileName(self, "Save File", filepath, "Campaign (*.qvt)")

            if (filepath == ""):
                filepath = None
            
            else:
                self.setRecentFile(filepath)

        if (filepath is None):
            return

        gscene = self.gscene

        self.showMessage("Saving %r" % filepath)
        self.campaign_filepath = filepath
        self.setWindowTitle("QtVTT - %s" % os.path.basename(filepath))

        logger.info("saving %r", filepath)
        logger.debug("%r", [attr for attr in self.scene.__dict__])

        # Use a set since eg token filepaths can be duplicated and want to save
        # them only once
        filepaths = set()
        pixmaps = dict()
        
        # Refresh door open/close
        # XXX This should be removed once scene changes are tracked properly in
        #     the model
        for door_item in gscene.doors():
            door = door_item.data(0)
            door.open = gscene.isDoorOpen(door_item)
        
        # Get the scene as a dict copy (need a copy since the fields are
        # modified below and don't want to modify the original scene with
        # json-friendly version of the fields)
        # XXX This should be recursive so it doesn't need to be done explicitly
        #     below? or have a json formatter that understands Struct
        d = dict(vars(self.scene))

        # Collect music as dicts instead of Structs
        filepaths |= set([track.filepath for track in d.get("music", [])])
        d["music"] = [vars(track) for track in d.get("music", [])]

        # Collect doors as dicts instead of Structs
        d["map_doors"] = [vars(door) for door in d["map_doors"]]

        # XXX Store coords in separate file/buffer glb style?

        # XXX Add some other scene information like scene name, first saved,
        #     last saved, number of saves
        
        # XXX Add some campaign information like campaign name, first saved,
        #     last saved, number of saves
        
        # XXX Could keep backup copies / diffs in some entry, is there a way of
        #     doing that reusing the existing zip? Would need to delete the file
        #     and copy over anyway when going over the backup limit since zipfiles
        #     can't delete entries
        
        # XXX Keep backup copies by keeping counter-suffixed zip files or even
        #     zips inside zips?

        # Add tokens, note token_items is a set, sort by path for good measure
        # so eg saving the same file twice is guaranteed to match (note
        # duplicated paths are removed so it cannot happen that two items
        # compare the same and they cause different savefiles because they
        # happened to sort differently in one save vs the other)
        # XXX Change this to use the model directly once this is recycled for the
        #     GraphicsScene to model cases
        tokens = [ 
            { 
                "filepath" : os.path.relpath(gscene.getTokenPixmapItem(token_item).data(0).filepath), 
                "scene_pos" : qtuple(gscene.getTokenPixmapItem(token_item).scenePos()),
                # Note this stores a resolution independent scaling, it has to
                # be divided by the width at load time
                # XXX This assumes the scaling preserves the aspect ratio, may 
                #     need to store scalex and scaly
                "scale" : gscene.getTokenPixmapItem(token_item).scale() * gscene.getTokenPixmapItem(token_item).pixmap().width(),
                "name" :  gscene.getTokenLabelItem(token_item).toPlainText(),
                "hidden" : gscene.getTokenPixmapItem(token_item).data(0).hidden,
                "ruleset_info" : getattr(gscene.getTokenPixmapItem(token_item).data(0), "ruleset_info", Struct(**default_ruleset_info))
            }  for token_item in sorted(gscene.tokens(), cmp=lambda a, b: cmp(a.data(0).filepath, b.data(0).filepath))
        ]
        d["map_tokens"] = tokens
        pixmaps.update({ gscene.getTokenPixmapItem(token_item).data(0).filepath : gscene.getTokenPixmapItem(token_item).pixmap() for token_item in gscene.tokens()})

        images = [
            {
                "filepath" :  os.path.relpath(image_item.data(0).filepath), 
                "scene_pos" : qtuple(image_item.scenePos()),
                # Note this stores a resolution independent scaling, it has to
                # be divided by the width at load time
                # XXX This assumes the scaling preserves the aspect ratio, may 
                #     need to store scalex and scaly
                "scale" : image_item.scale() * image_item.pixmap().width()
            } for image_item in sorted(gscene.images(), cmp=lambda a, b: cmp(a.data(0).filepath, b.data(0).filepath))
        ]
        d["map_images"] = images
        pixmaps.update({ image_item.data(0).filepath : image_item.pixmap() for image_item in gscene.images()})
        
        # XXX This should append to whatever scenes or store each scene
        #     in a directory?
        d = { "version" : 1.0, "scenes" : [d] }

        # XXX Save the focused player, enabled fog blend etc? (which of those 
        #     should be app settings and which scene settings?)

        # XXX This should append if the file exists, preserving the existing
        #     files and updating only the new ones (not clear what to do if
        #     an image was updated, is it ok to update for all the references?)
        #     Will also need to delete the archived files if longer present
        
        # Note zipfile by default will keep adding files with the same name
        # without deleting the old ones. Zip to a temp and then copy over (this
        # is also safer in the presence of errors)
        tmp_filepath = "%s~" % filepath
        orig_filepath = filepath
        with zipfile.ZipFile(tmp_filepath, "w") as f:
            f.writestr("campaign.json", json.dumps(d, indent=2, cls=JSONStructEncoder))
            
            # Zip and store the tokens, images
            # XXX Should probably recreate the fullpaths or create some
            #     uid out of it in case of collision, but would also be 
            #     nice to not duplicate assets shared across several scenes
            #     if a campaign is saved
            #     Use filename + uid?
            
            # XXX Embedding the assets should be optional, eg if it's actively
            #     editing the map in an external tool it's not productive to
            #     have to manually re-import/update the zip file everytime
            # XXX Could do embedded if .qvt, non-embedded if .json
            
            # XXX Have an option to embed the assets as data urls
            
            # For pixmaps, the images may have been downscaled when loaded,
            # store whatever the pixmap contains
            
            # XXX Disabled until assets are loaded from the file, and .json or
            #     .json.gz are used for non embedded assets, and the path issues in
            #     the zip file are dealt with, and the recompression is dealt
            #     with
            if (False):
                for filepath, pixmap in pixmaps.iteritems():
                    ba = QByteArray()
                    buff = QBuffer(ba)
                    buff.open(QIODevice.WriteOnly) 
                    
                    _, ext = os.path.splitext(filepath)
                    fmt = ext[1:].upper()
                    # XXX This is recompressing, should keep the existing file 
                    # XXX This assumes that all the assets with the same filepath
                    #     have the same downscaling

                    # XXX Should this store the original file instead? (the stored
                    #     scale factor is already resolution independent so that's
                    #     not an issue)
                    ok = pixmap.save(buff, fmt)
                    assert ok
                    
                    f.writestr(filepath, ba.data())

                for filepath in filepaths:
                    f.write(filepath)

        # Note rename raises on Windows if already exists, delete and rename
        if (os.path.exists(orig_filepath)):
            os.remove(orig_filepath)
        os.rename(tmp_filepath, orig_filepath)

        self.showMessage("Saved %r" % orig_filepath)

    def saveSceneAs(self):
        filepath = self.campaign_filepath
        if (filepath is None):
            #dirpath = os.path.curdir if filepath is None else filepath
            filepath = os.path.join("_out", "campaign.qvt")
        filepath, _ = QFileDialog.getSaveFileName(self, "Save File", filepath, "Campaign (*.qvt)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        self.setRecentFile(filepath)

        self.campaign_filepath = filepath
        self.setWindowTitle("QtVTT - %s" % os.path.basename(filepath))
        self.saveScene()

    def addTokensFromEncounter(self, builder):
        scene = self.scene
        # Unselect all items, will leave added items as selected so they can be
        # easily moved/deleted
        self.gscene.clearSelection()
        for j in xrange(builder.encounterTable.rowCount()):
            ruleset_info = dict()

            for header, i in builder.encounterHeaderToColumn.iteritems():
                item = builder.encounterTable.item(j, i)
                # XXX Stop using chars non valid for struct fields?
                if (header == "#AT"):
                    header = "AT"
                cell = item.text()
                ruleset_info[header] = cell

            dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
            # XXX Get supported extensions from Qt
            filepath, _ = QFileDialog.getOpenFileName(self, "Import Token for '%s'" % ruleset_info["Name"], dirpath, "Images (*.png *.jpg *.jpeg *.jfif *.webp)")

            if ((filepath == "") or (filepath is None)):
                # XXX Should this abort adding tokens? What about the already added?
                filepath = os.path.join("_out", "tokens", "knight.png")

            name = ruleset_info["Name"]
            
            del ruleset_info["Name"]
            
            ruleset_info = Struct(**ruleset_info)

            # Center on viewport, offset each token half a token size
            # XXX Do some other layout (grid?)
            scene_pos = qtuple(self.graphicsView.mapToScene(QPoint(j, j) * scene.cell_diameter + qSizeToPoint(self.graphicsView.size()/2.0)))
            map_token = Struct(**{
                "filepath" : os.path.relpath(filepath),
                "scene_pos" :  scene_pos,
                "name" :  name,
                "hidden" : False,
                "scale" : float(scene.cell_diameter),
                "ruleset_info" : ruleset_info
            })
        
            scene.map_tokens.append(map_token)
            # XXX Looks like this causes lots of updates (several seconds) even
            #     with a single monster in the encounter table, probably because
            #     of re-adding the existing tokens? find where to lock updates?
            token = self.gscene.addToken(map_token)
            # Select the pasted item
            token.setSelected(True)

        # XXX Use something less heavy handed than setScene
        # self.setScene(scene)
    
    def populateTokens(self, gscene, scene):
        for map_token in scene.map_tokens:
            gscene.addToken(map_token)
            
    def populateImages(self, gscene, scene):
        for map_image in scene.map_images:
            gscene.addImage(map_image)
            
    def populateWalls(self, gscene, scene):
        for map_wall in scene.map_walls:
            gscene.addWall(map_wall)
            
    def populateDoors(self, gscene, scene):
        for map_door in scene.map_doors:
            gscene.addDoor(map_door)

    def populateGrid(self, gscene, scene):
        gscene.addGrid(scene.cell_diameter)

    def populateGraphicsScene(self, gscene, scene):

        gscene.setCellDiameter(scene.cell_diameter)

        # Populated in z-order
        # XXX Fix, no longer the case since many groups are created beforehand
        self.populateImages(gscene, scene)
        
        self.populateGrid(gscene, scene)

        self.populateWalls(gscene, scene)

        self.populateDoors(gscene, scene)

        self.populateTokens(gscene, scene)

        # Set the rect, needs to be done before adding the fog since they add
        # very large polygons which would explode the bounding rect
        # unnecessarily
        rect = gscene.itemsBoundingRect()
        
        gscene.setSceneRect(rect)
        
    def generateSVG(self):
        # XXX updateFog generates fog_polys which is needed by generateSVG, but
        #     should use something lighter since it also generates the
        #     QGraphicScene items
        self.updateFog(True, False)
        # The amount of svg to render can be:
        # - svg: PC and NPC tokens, map, and fog are all rendered using svg.
        #   This uses a lot less load since only fog polygons and token
        #   positions need to be refreshed, but it's insecure since the clear
        #   map is sent, also empirically pinch scaling on the browser can show
        #   slivers of the map.
        # - svg with fog as image mask: Only requires sending an image of the
        #   fog on updates, which will also compress better and even be sent 
        #   donscaled. Still insecure.
        # - svg with fog and map as image: player tokens are svg, map and fog
        #   are merged into a single image. This increases the load because the
        #   image needs to be sent whenever NPC tokens or player tokens change
        gscene = self.gscene
        # XXX Use svg_bytes instead of saving to files
        with open(os.path.join("_out", "fog.svg"), "w") as f:
            sceneRect = gscene.sceneRect()
            f.write('<svg viewBox="%f %f %f %f" preserveAspectRatio="xMidYMid meet" onload="makeDraggable(evt)" xmlns="http://www.w3.org/2000/svg">\n' % 
                (sceneRect.left(), sceneRect.top(), 
                sceneRect.width(), sceneRect.height()))

            # See https://raw.githubusercontent.com/petercollingridge/code-for-blog/master/svg-interaction/draggable/draggable_groups.svg
            f.write("""
                <style>
                .static {
                    cursor: not-allowed;
                }
                .draggable, .draggable-group {
                    cursor: move;
                }
                </style>
                
                <script type="text/javascript"><![CDATA[
                function makeDraggable(evt) {
                    var svg = evt.target;

                    svg.addEventListener('mousedown', startDrag);
                    svg.addEventListener('mousemove', drag);
                    svg.addEventListener('mouseup', endDrag);
                    svg.addEventListener('mouseleave', endDrag);
                    svg.addEventListener('touchstart', startDrag);
                    svg.addEventListener('touchmove', drag);
                    svg.addEventListener('touchend', endDrag);
                    svg.addEventListener('touchleave', endDrag);
                    svg.addEventListener('touchcancel', endDrag);

                    function getMousePosition(evt) {
                    var CTM = svg.getScreenCTM();
                    if (evt.touches) { evt = evt.touches[0]; }
                    return {
                        x: (evt.clientX - CTM.e) / CTM.a,
                        y: (evt.clientY - CTM.f) / CTM.d
                    };
                    }

                    var selectedElement, offset, transform;

                    function initialiseDragging(evt) {
                        offset = getMousePosition(evt);

                        // Make sure the first transform on the element is a translate transform
                        var transforms = selectedElement.transform.baseVal;

                        if (transforms.length === 0 || transforms.getItem(0).type !== SVGTransform.SVG_TRANSFORM_TRANSLATE) {
                        // Create an transform that translates by (0, 0)
                        var translate = svg.createSVGTransform();
                        translate.setTranslate(0, 0);
                        selectedElement.transform.baseVal.insertItemBefore(translate, 0);
                        }

                        // Get initial translation
                        transform = transforms.getItem(0);
                        offset.x -= transform.matrix.e;
                        offset.y -= transform.matrix.f;
                    }

                    function startDrag(evt) {
                    if (evt.target.classList.contains('draggable')) {
                        selectedElement = evt.target;
                        initialiseDragging(evt);
                    } else if (evt.target.parentNode.classList.contains('draggable-group')) {
                        selectedElement = evt.target.parentNode;
                        initialiseDragging(evt);
                    }
                    }

                    function drag(evt) {
                    if (selectedElement) {
                        evt.preventDefault();
                        var coord = getMousePosition(evt);
                        transform.setTranslate(coord.x - offset.x, coord.y - offset.y);
                    }
                    }

                    function endDrag(evt) {
                    selectedElement = false;
                    }
                }
                ]]> </script>
            """)
                    
            sceneRect = list(self.gscene.images())[0].sceneBoundingRect()
            f.write('<image href="image.png" x="%f" y="%f" width="%f" height="%f"/>\n' %
                (sceneRect.x(), sceneRect.top(), sceneRect.width(), sceneRect.height()))

            for token_item in self.gscene.tokens():
                sceneRect = token_item.sceneBoundingRect()

                imformat = "PNG"
                ba = QByteArray()
                buff = QBuffer(ba)
                buff.open(QIODevice.WriteOnly) 
                ok = token_item.pixmap().save(buff, imformat)
                assert ok
                img_bytes = ba.data()
                import base64

                base64_utf8_str = base64.b64encode(img_bytes).decode('utf-8')

                dataurl = 'data:image/png;base64,%s' % base64_utf8_str
                label_item = token_item.childItems()[0]

                f.write('<g class="draggable-group" transform="translate(%f,%f) scale(%f)"><image href="%s" width="%f" height="%f"/><text fill="white" font-size="10" font-family="Arial, Helvetica, sans-serif" transform="translate(%f,%f) scale(%f)">%s</text></g>\n' %
                    (sceneRect.x(), sceneRect.top(), token_item.scale(), dataurl, token_item.pixmap().width(), token_item.pixmap().height(), 
                        label_item.pos().x(), label_item.pos().y(), label_item.scale(), label_item.toPlainText()))

                f.write('')

            for poly in self.fog_polys:
                poly_string = ["%f,%f" % point for point in poly]
                poly_string = str.join(" ", poly_string)

                # <polygon points="0,100 50,25 50,75 100,0" />
                #s = '<polygon fill="none" stroke="black" points="%s"/>\n' % poly_string
                s = '<polygon fill="black" stroke="black" points="%s"/>\n' % poly_string

                f.write(s)

            f.write("</svg>\n")


    def updateFog(self, draw_map_fog, blend_map_fog):
        if (self.gscene.getFogCenter() is None):
            logger.warning("Called updateFog with no token item!!")
            return

        self.gscene.setFogVisible(draw_map_fog)
        self.gscene.setBlendFog(blend_map_fog)
        # Fog already updated by setFogVisible and setBlendFog
        # XXX Missing updating these
        self.fog_polys = []
        
    def updateImage(self):
        logger.info("updateImage")
        global img_bytes
        gscene = self.gscene
        fogCenter = gscene.getFogCenter()
        img_scale = 1.0

        gscene.setLockDirtyCount(True)

        # XXX May want to gscene.clearselection but note that it will need to be
        #     restored afterwards or the focus rectangle will be lost in the DM
        #     view when moving, switching to a different token, etc
        
        # Grow to 64x64 in case there's no scene
        qim = QImage(gscene.sceneRect().size().toSize().expandedTo(QSize(1, 1)) * img_scale, QImage.Format_ARGB32)
        # If there's no current token, there's no fog, just clear to background
        # and don't leak the map
        if (fogCenter is None):
            qim.fill(QColor(196, 196, 196))

        else:
            use_svg = False
            if (use_svg):
                self.generateSVG()
                # Use the unfogged image for svg
                # XXX Implement svg fog, image fog or image fog + map
                pix = gscene.imageAt(0).pixmap()
                qim = QImage(pix)
            
            else:
                
                self.updateFog(True, False)

                # Hide all DM user interface helpers
                # XXX Hiding seems to be slow, verify? Try changing all to transparent
                #     otherwise? Have a player and a DM scene?
                logger.info("hiding DM ui")
                gscene.setWallsVisible(False)
                gscene.setDoorsVisible(False)
                gscene.setTokensHiddenFromPlayerVisible(False)
                logger.info("Rendering %d scene items on %dx%d image", len(gscene.items()), qim.width(), qim.height())
                p = QPainter(qim)
                gscene.render(p)
                # This is necessary so the painter winds down before the pixmap
                # below, otherwise it crashes with "painter being destroyed
                # while in use"
                p.end()
                
                # Restore all DM user interface helpers
                logger.info("restoring DM ui")
                gscene.setWallsVisible(self.graphicsView.drawWalls)
                gscene.setDoorsVisible(True)
                gscene.setTokensHiddenFromPlayerVisible(True)

                self.updateFog(self.graphicsView.drawMapFog, self.graphicsView.blendMapFog)
                
            
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

        if (fogCenter is not None):
            self.imageWidget.ensureVisible(
                # Zero-based position of the item's center, but in pixmap
                # coordinates
                ((fogCenter.x() - gscene.sceneRect().x()) * img_scale) * self.imageWidget.scale, 
                ((fogCenter.y() - gscene.sceneRect().y()) * img_scale) * self.imageWidget.scale,
                self.imageWidget.width() / 4.0, 
                self.imageWidget.height() / 4.0
            )

        gscene.setLockDirtyCount(False)
        
    def sceneChanged(self, region):
        """
        By default sceneChanged sends changes for all non-programmatic changes
        in the scene (eg dragging an item sends but calling setpos doesn't)

        Ideally would hook on sceneChanged to update the fog item and the player
        image whenever something in the scene changes, but that causes infinite
        calls since fog item changes causes sceneChanged ad infinitum
        (sceneChanged gets a raw changed region so it can't filter out specific
        items).

        Checking a reentrant flag in order to prevent infinite recursion into
        sceneChanged is not good enough because it appears sceneChanged is
        called asynchronously wrt to the changes, ie the changes are batched
        when they happen and sceneChanged called from some scene idle loop (as
        described in Qt documentation).

        There's another notification method which is the per-item itemChanged
        event, but in order to use it the item flags
        ItemSendsScenePositionChanges needs to be set, the item subclassed and
        itemChanged even handler overriden. By not overriding/setting the flag
        for the fog items, the fog modification can be filtered out.

        In the current implementation, the itemChanged event is forwarded to a
        new scene.itemChanged, which increments a scene dirty count so the next
        time sceneChanged is called, the dirty count is checked and fog and
        image updated. When the fog is updated, there's no dirty count increased
        which prevents the infinite change.

        On a scene change, need to update:
        - the image
        - the fog
        - the graphicsview

        Cases:
        - a door is opened with the keyboard/mouse:
            - gscene.makeDirty()
        - a token is moved with the keyboard/hidden/etc:
            - gscene.makeDirty()
            - moving the token with the keyboard calls setpos, which doesn't
              call itemchanged but it does call scenechanged once the changes
              are processed
            - scenechanged seees the scene dirty and calls updateimage, which
              also sees the fog dirty and recalculates the fog and the image
        - a token is moved with the mouse:
            - nothing
            - moving the token with the mouse calls itemchanged, which sets the
              dirty flag and calls scenechanged as above
        - fog settings change (blend, enable, no change to fog polys)
            - updateFog()
            - fog polys are still correct, so the fog is re-rendered with the
              same polys, which causes scene updates but are ignored because the
              dirty count stays the same
        - fogcenter is unlocked (change to fog polys)
            - gscene.makeDirty() & gscene.invalidate()
            - Need to dirty and then cause a full invalidation, which triggers
              sceneChange (because items didn't change position, there's no call
              to itemChanged, hence the need to dirty beforehand)
        - fog is updated inside updateimage 
            - gscene.lock & ... & update fog & gscene.unlock
            XXX Not clear the lock is doing anything because of the batching?
            XXX Use blocksignals for this
        """
        logger.info("sceneChanged")
        gscene = self.gscene
        if (gscene.dirtyCount != self.sceneDirtyCount):
            self.sceneDirtyCount = gscene.dirtyCount
            # XXX These should go via signals
            # updateImage calls updateFog, always renders polys but only
            # recalcualtes fog if fog dirty count mismatches scene dirty count
            self.updateImage()
            # XXX This is too heavy-handed, since it resets the tree focus and
            #     fold state
            self.updateTree()
            self.updateCombatTrackers()
            # XXX Hook here the scene undo stack? (probably too noise unless
            #     filtered by time or by diffing the scene?)

            
            fogCenter = gscene.getFogCenter()
            if (fogCenter is not None):
                fogCenter = fogCenter.x(), fogCenter.y()
                name = "????"
                if (gscene.getFogCenterLocked()):
                    name = "LOCKED"
                    
                elif (gscene.isToken(gscene.focusItem())):
                    map_token = gscene.focusItem().data(0)
                    name = map_token.name
                    if (map_token.hidden):
                        name = "*" + name
                    
                s = "%s: %.01f,%.01f " % (name, fogCenter[0], fogCenter[1])

            else:
                s = ""

            # XXX Assumes 5ft per cell
            s += " LR %.2f" % (gscene.getLightRange() * 5.0 / gscene.getCellDiameter())
            s += " S" if (gscene.getSnapToGrid()) else " NS"
            s += " G" if (self.graphicsView.drawGrid) else " NG"
                
            self.statusScene.setToolTip("%d walls, %d doors, %d images, %d tokens" % (
                len(self.scene.map_walls),
                len(self.scene.map_doors),
                len(self.scene.map_images),
                len(self.scene.map_tokens)
            ))
            self.statusScene.setText(s)
    
    def eventFilter(self, source, event):
        logger.debug("source %r type %s", class_name(source), qEventTypeToString(event.type()))
        
        if ((event.type() == QEvent.KeyPress) and (source is self.imageWidget)):
            logger.info("text %r key %d", event.text(), event.key())
            if (event.text() == "f"):
                self.imageWidget.setFitToWindow(not self.imageWidget.fitToWindow)
                self.imageWidget.update()
        
            elif (event.text() == "x"):
                # Toggle maximize / restore dock
                dock = self.imageWidget.parent()
                if (dock.isFloating()):
                    dock.setFloating(False)
                    # Imagewidget loses focus when docking, bring it back
                    self.imageWidget.setFocus(Qt.TabFocusReason)
                    self.imageWidget.setFitToWindow(True)

                else:
                    dock.setFloating(True)
                    dock.setWindowState(Qt.WindowMaximized)
                    self.imageWidget.setFitToWindow(True)
            
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