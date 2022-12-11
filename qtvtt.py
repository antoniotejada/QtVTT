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

    logger.info("os_path_abspath %r is %r", path, abspath)
    return abspath


def index_of(l, item):
    """
    Find the first occurrence of item in the list or return -1 if not found
    """
    try:
        return l.index(item)

    except ValueError:
        return -1


def qtuple(q):
    """
    Convert Qt vectors (QPoint, etc) to tuples
    """
    if (isinstance(q, (QPoint, QPointF))):
        return (q.x(), q.y())

    elif (isinstance(q, (QSize, QSizeF))):
        return (q.width(), q.height())

    elif (isinstance(q, (QRect, QRectF))):
        return (qtuple(q.p1()), qtuple(q.p2()))
    else:
        assert False, "Unhandled Qt type!!!"


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
                    map_doors.append(Struct(lines=door, open=False))
                
        # These correspond to the door handles, adding as walls
        # XXX Do a better parse that doesn't assume two lines per entry?
        map_walls.extend([line[0] + line[1] for line in lines])

    logger.info("Found %d doors %d valid walls, %d coalesced, %d duplicated", len(map_doors), len(map_walls), 
        coalesced_walls, duplicated_walls)
    
    scene.cell_diameter = map_cell_diameter
    scene.map_walls = map_walls
    scene.map_doors = map_doors

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
            img_offset_in_cells = [round(c) for c in qtuple(-(bounds.topLeft() - QPointF(margin.width(), margin.height())) / scene.cell_diameter)]

        scene.map_images = [
            Struct(**{
                # XXX Some dungeon scrawl v1 png files are saved with the wrong pixel
                #     height, ideally should do rounding at png loading time?
                "scale" : scene.cell_diameter * img_size_in_cells[0],
                "filepath" : map_filepath,
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
        
        pix_scale = scene.cell_diameter
        # XXX This should check the monster size or such
        if ("Dragon" in filename):
            pix_scale *= 4.0
        elif ("Ogre" in filename):
            pix_scale *= 1.5

        map_token.scale = pix_scale

        scene.map_tokens.append(map_token)    

    return scene


def build_index():
    logger.info("building index")
    dirpath = R"..\..\jscript\vtt\_out"
    words = dict()
    filepath_to_title = dict()
    title_to_filepath = dict()
    for subdirpath in ["Monsters1", R"cdrom\WEBHELP\DMG", R"cdrom\WEBHELP\PHB"]:
        print "indexing", subdirpath
        for filename in os.listdir(os.path.join(dirpath, subdirpath)):
            if (filename.lower().endswith((".htm", ".html"))):
                with open(os.path.join(dirpath, subdirpath, filename), "r") as f:
                    print "reading", filename

                    subfilepath = os.path.join(subdirpath, filename)

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
                    # XXX Use some Qt class to remove HTML tags?
                    for word in re.split(r"\W+", s):
                        if (word != ""):
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



class DocBrowser(QWidget):
    """
    Documentation browser with HTML browser, filtering/searching, filter/search
    results list, filter/search results list and hit navigation and table of
    contents tree
    """
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

        indexFilepath = os.path.join("_out", "index.json")
        if (not os.path.exists(indexFilepath)):
            build_index()

        with open(indexFilepath, "r") as f:
            js = json.load(f)

        index = Struct()
        index.word_to_filepaths = js["word_to_filepaths"]
        index.title_to_filepath = js["title_to_filepath"]
        index.filepath_to_title = js["filepath_to_title"]
        
        keys = index.word_to_filepaths.keys()
        # Convert from dict of lists to dict of sets, use lowercase as key to
        # make it case-insensitive
        for key in keys:
            s = set(index.word_to_filepaths[key])
            del index.word_to_filepaths[key]
            index.word_to_filepaths[key.lower()] = set(index.word_to_filepaths.get(key.lower(), [])) | s

        lineEdit = QLineEdit(self)
        listWidget = QListWidget()
        tocTree = QTreeWidget()
        tocTree.setColumnCount(1)
        tocTree.setHeaderHidden(True)
        
        self.lastCursor = None
        self.lastPattern = None
        self.curTocFilePath = None
        self.curTocItem = None
        self.index = index
        
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
        #textEdit.setSource(QUrl.fromLocalFile(R"..\..\jscript\vtt\_out\cdrom\WEBHELP\MM\DD03846.HTM"))
        #textEdit.setSource(QUrl.fromLocalFile(R"..\..\jscript\vtt\_out\Monsters1\MM00057.htm"))
        #textEdit.setSource(QUrl.fromLocalFile(R"..\..\jscript\vtt\_out\monstrousmanual\d\dragcgre.html"))
        #textEdit.setSource(QUrl.fromLocalFile(R"..\..\jscript\vtt\_out\mm\dragcred.html"))
        
        
        hsplitter = QSplitter(Qt.Horizontal)
        hsplitter.addWidget(tocTree)
        hsplitter.addWidget(textEdit)
        hsplitter.setStretchFactor(1, 20)
        
        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(listWidget)
        vsplitter.addWidget(hsplitter)
        vsplitter.setStretchFactor(1, 10)

        vbox = QVBoxLayout()
        vbox.addWidget(lineEdit, 0)
        vbox.addWidget(vsplitter)

        self.setLayout(vbox)        

        return 

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

        self.listWidget.clear()
        if (len(items) == 0):
            # Empty results, reload the current document so marks are cleared
            self.textEdit.setSource(self.textEdit.source())

        else:
            self.listWidget.addItems(items)
            logger.info("Going to %d item", 0)
            self.listWidget.setCurrentRow(0)

    def treeCurrentItemChanged(self, current, previous):
        if (current is None):
            return

        filepath = current.data(Qt.UserRole, 0)
        filepath = os_path_normall(filepath)
        urlpath = os_path_normall(self.textEdit.source().path())

        logger.info("Sync browser to tree %s", filepath)
        if (urlpath != filepath):
            url = QUrl.fromLocalFile(filepath)
            logger.info("Setting source from %r to %r", urlpath, url)
            self.textEdit.setSource(url)
            
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
                with open(os.path.join(R"..\..\jscript\vtt\_out", tocFilepath), "r") as f:
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
                            parentItem.setData(Qt.UserRole, 0, tocFilepath)
                            
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

                        curItem.setData(Qt.UserRole, 0, os.path.join(os.path.dirname(filepath), entryHref))

            # Activate the item for this filepath
            itemStack = [self.tocTree.topLevelItem(0)]
            nfilepath = os_path_normall(filepath)
            while (len(itemStack) > 0):
                item = itemStack.pop()
                # Note hrefs case may mismatch from filepaths case and from
                # XXX Fix outside of the loop?

                nitempath = os_path_normall(item.data(Qt.UserRole, 0))
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

        self.textEdit.setSource(
            QUrl.fromLocalFile(os.path.join(R"..\..\jscript\vtt\_out", filepath)))

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

    # XXX In case some computation is needed between requests, this can also do
    #     httpd.handle_request in a loop
    httpd.serve_forever()
    
g_draw_walls = True
g_blend_map_fog = False
g_draw_map_fog = False
g_draw_grid = True
most_recently_used_max_count = 10

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

class VTTMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(VTTMainWindow, self).__init__(parent)

        self.campaign_filepath = None
        self.recent_filepaths = []

        self.gscene = None
        self.scene = None

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

        view = QGraphicsView()
        view.installEventFilter(self)
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView = view

        tree = QTreeWidget()
        self.tree = tree

        self.browser = DocBrowser()

        self.docks = [
            (QDockWidget("Player View", self), self.imageWidget),
            (QDockWidget("DM View", self), self.graphicsView),
            (QDockWidget("Campaign", self), self.tree),
            (QDockWidget("Browser", self), self.browser)
        ]
        for dock, view in self.docks:
            # Set the object name, it's necessary so Qt can save and restore the 
            # state in settings
            dock.setObjectName(dock.windowTitle())
            dock.setWidget(view)
            dock.setFloating(False)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        for dock, view in self.docks:
            self.addDockWidget(Qt.TopDockWidgetArea, dock)

        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)

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
            if (not filepath):
                break
            self.recent_filepaths.append(filepath)
        # Set the first as most recent, will keep all entries the same but will
        # update all the menus
        if (len(self.recent_filepaths) > 0):
            self.setRecentFile(self.recent_filepaths[0])
            
        # Set the scene, which will update graphicsview, tree, imagewidget
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

        logger.info("Restoring window geometry and state")
        settings.beginGroup("layout")
        b = settings.value("geometry")
        if (b):
            self.restoreGeometry(b)
        b = settings.value("windowState")
        if (b):
            self.restoreState(b)
        settings.endGroup()

    def createMusicPlayer(self):
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


    def createActions(self):
        self.newAct = QAction("&New...", self, shortcut="ctrl+n", triggered=self.newScene)
        self.openAct = QAction("&Open...", self, shortcut="ctrl+o", triggered=self.openScene)
        self.importDsAct = QAction("Import &Dungeon Scrawl...", self, triggered=self.importDs)
        self.saveAct = QAction("&Save", self, shortcut="ctrl+s", triggered=self.saveScene)
        self.saveAsAct = QAction("Save &As...", self, shortcut="ctrl+shift+s", triggered=self.saveSceneAs)
        self.exitAct = QAction("E&xit", self, shortcut="alt+f4", triggered=self.close)

        self.recentFileActs = []
        for i in range(most_recently_used_max_count):
            self.recentFileActs.append(
                    QAction(self, visible=False, triggered=self.openRecentFile))

        self.deleteItemAct = QAction("&Delete Item", self, shortcut="del", triggered=self.deleteItem)
        self.importTokenAct = QAction("Import &Token", self, shortcut="ctrl+T", triggered=self.importToken)
        self.importImageAct = QAction("Import &Image", self, shortcut="ctrl+I", triggered=self.importImage)
        self.clearWallsAct = QAction("Clear All &Walls", self, triggered=self.clearWalls)
        self.clearDoorsAct = QAction("Clear All &Doors", self, triggered=self.clearDoors)
        self.copyScreenshotAct = QAction("DM &View &Screenshot", self, shortcut="ctrl+c", triggered=self.copyScreenshot)
        self.copyFullScreenshotAct = QAction("DM &Full Screenshot", self, shortcut="ctrl+shift+c", triggered=self.copyFullScreenshot)

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
        editMenu.addSeparator()
        editMenu.addAction(self.deleteItemAct)
        editMenu.addSeparator()
        editMenu.addAction(self.clearDoorsAct)
        editMenu.addAction(self.clearWallsAct)
        editMenu.addSeparator()
        editMenu.addAction(self.copyScreenshotAct)
        editMenu.addAction(self.copyFullScreenshotAct)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.aboutAct)
        helpMenu.addAction(self.aboutQtAct)

        bar = self.menuBar()
        bar.addMenu(fileMenu)
        bar.addMenu(editMenu)
        bar.addSeparator()
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

        self.statusFilepath = QLabel()
        self.statusFilepath.setFrameStyle(frame_style)
        self.statusBar().addPermanentWidget(self.statusFilepath)


    def about(self):
        QMessageBox.about(self, "About QtVTT",
                "<p>Simple no-frills <b>virtual table top</b>:"
                "<ul>"
                "<li> token collision "
                "<li> line of sight"
                "<li> remote viewing via http"
                "<li> <a href=\"https://app.dungeonscrawl.com/\">Dungeon Scrawl</a> import"
                "<li> and more"
                "</ul>"
                "Visit the <a href=\"https://github.com/antoniotejada/QtVTT\">github repo</a> for more information</p>")
                

    def closeEvent(self, e):
        logger.info("closeEvent")
        
        # XXX May want to have a default and then a per-project setting 
        # XXX May want to have per resolution settings
        settings = self.settings

        # XXX Should also save and restore the zoom positions, scrollbars, tree
        # folding state

        logger.info("Storing window geometry and state")
        settings.beginGroup("layout")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.endGroup()

        logger.info("Storing most recently used")
        for i, filepath in enumerate(self.recent_filepaths):
            settings.setValue("recentFilepath%d" %i, filepath)
        
        e.accept()
        
    def showMessage(self, msg, timeout_ms=2000):
        self.status_message_timer.stop()
        self.status_message_widget.setText(msg)
        if (timeout_ms > 0):
            self.status_message_timer.start(timeout_ms)
            
    def clearMessage(self):
        self.status_message_widget.setText("")

    def setScene(self, scene, filepath = None):
        logger.info("setScene")

        print_gc_stats()

        self.scene = scene

        # XXX Verify if all this old gscene cleaning up is necessary and/or
        #     enough
        if (self.gscene is not None):
            self.gscene.clear()
            self.gscene.setParent(None)
        self.map_fog_item = None
        self.fog_polys = []

        gscene = QGraphicsScene()
        # XXX It's not clear the BSP is helping on dynamic scenes with fog
        #     (although the fog is not selectable so it shouldn't be put in the
        #     bsp?)
        ##gscene.setItemIndexMethod(QGraphicsScene.NoIndex)

        self.campaign_filepath = filepath
        if (filepath is not None):
            self.setWindowTitle("QtVTT - %s" % os.path.basename(filepath))

        else:
            self.setWindowTitle("QtVTT")

        self.populateGraphicsScene(gscene, scene)
        self.graphicsView.setScene(gscene)
        # XXX Can this filter be set at the view level so it doesn't need to be
        #     reset when reloading a scene?
        gscene.installEventFilter(self)

        self.gscene = gscene

        self.updateTree()
        self.updateImage()

        # Repaint the image widget and start with some sane scroll defaults
        # XXX Note this will reset the scroll when clearing all walls/doors since
        #     they call setScene for now
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

        self.setScene(scene)

    
    def importToken(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Token", dirpath, "Images (*.png *.jpg *.jpeg *.jfif *.webp)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return
        
        s = Struct(**{
            # Center the token in the current viewport
            "scene_pos": 
                qtuple(self.graphicsView.mapToScene(QPoint(*qtuple(self.graphicsView.size()/2.0))))
            , 
            # XXX Fix all float casting this malarkey, cast it in whatever
            #     operation needs it, not in the input data
            "scale": float(self.scene.cell_diameter), 
            "name": os.path.splitext(os.path.basename(filepath))[0], 
            # XXX Fix all the path mess for embedded assets
            "filepath": os.path.relpath(filepath)
        })
        self.scene.map_tokens.append(s)
        # XXX Use something less heavy handed than setScene
        self.setScene(self.scene, self.campaign_filepath)

    def importImage(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
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
        
    def deleteItem(self):
        # XXX This only expects tokens for the time being

        # XXX This should disable the menu once hooked on focuschange
        if (self.gscene.focusItem() is None):
            return

        self.scene.map_tokens.remove(self.gscene.focusItem().data(1))
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

        
    def importDs(self):
        # Get the ds filename
        dirpath = os.path.curdir if self.campaign_filepath is None else self.campaign_filepath
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Dungeon Scrawl data file", dirpath, "Dungeon Scrawl (*.ds)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        ds_filepath = filepath

        # Try to load a similarly name png, otherwise ask for the png filename
        l = os.listdir(os.path.dirname(ds_filepath))
        map_filepath = None
        pattern = os.path.splitext(os.path.basename(ds_filepath))[0]
        for filename in l:
            if (filename.startswith(pattern) and filename.endswith(".png")):
                map_filepath = os.path.join(os.path.dirname(ds_filepath), filename)
                break

        else:
        
            dirpath = os.path.curdir if self.campaign_filepath is None else self.campaign_filepath
            filepath, _ = QFileDialog.getOpenFileName(self, "Import Dungeon Scrawl battlemap", dirpath, "PNG file (*.png)")

            if (filepath == ""):
                filepath = None

            map_filepath = filepath

        # Note the map filepath can be none if they decided to load only walls 
        scene = load_ds(ds_filepath, map_filepath)
        # XXX Setting a filepath here skips the most recently used and won't ask
        #     for confirmation when saving, needs better flow
        filepath = os.path.splitext(ds_filepath)[0] + ".qvt"
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
        scene.map_images = [Struct(**map_image) for map_image in js["map_images"]]

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
        
        folder_item = QTreeWidgetItem(["Walls (%d)" % len(self.scene.map_walls)])
        scene_item.addChild(folder_item)
        for wall in self.scene.map_walls: 
            child = QTreeWidgetItem(["%s" % (wall,)])
            folder_item.addChild(child)
        
        folder_item = QTreeWidgetItem(["Doors (%d)" % len(self.scene.map_doors)])
        scene_item.addChild(folder_item)
        for door in self.scene.map_doors: 
            child = QTreeWidgetItem(["%s%s" % ("*" if door.open else "", door.lines)])
            folder_item.addChild(child)

        folder_item = QTreeWidgetItem(["Images (%d)" % len(self.scene.map_images)])
        scene_item.addChild(folder_item)
        for image in self.scene.map_images:
            subfolder_item = QTreeWidgetItem(["%s" % os.path.basename(image.filepath)])
            item = QTreeWidgetItem(["%s" % (image.scene_pos,)])
            subfolder_item.addChild(item)
            item = QTreeWidgetItem(["%s" % image.scale])
            subfolder_item.addChild(item)

            folder_item.addChild(subfolder_item)

        folder_item = QTreeWidgetItem(["Tokens (%d)" % len(self.scene.map_tokens)])
        scene_item.addChild(folder_item)
        for token in self.scene.map_tokens:
            subfolder_item = QTreeWidgetItem(["%s" % token.name])

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

        self.showMessage("Saving %r" % filepath)
        self.campaign_filepath = filepath
        self.setWindowTitle("QtVTT - %s" % os.path.basename(filepath))

        logger.info("saving %r", filepath)
        logger.debug("%r", [attr for attr in self.scene.__dict__])

        # Use a set since eg token filepaths can be duplicated and want
        # to save them only once
        filepaths = set()
        pixmaps = dict()
        
        # Refresh door open/close
        # XXX This should be removed once scene changes are tracked properly in
        #     the model
        for door_item in self.door_items:
            door = door_item.data(0)
            door.open = (door_item in self.open_door_items)
        
        # Get the scene as a dict copy (need a copy since the fields are
        # modified below and don't want to modify the original scene with
        # json-friendly version of the fields)
        # XXX This should be recursive so it doesn't need to be done explicitly
        #     below? or have a json formatter that understands Struct
        d = dict(vars(self.scene))

        # Collect doors as dicts instead of Structs
        d["map_doors"] = [vars(door) for door in d["map_doors"]]

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
                "filepath" : token_item.data(0), 
                "scene_pos" : qtuple(token_item.scenePos()),
                # Note this stores a resolution independent scaling, it has to
                # be divided by the width at load time
                # XXX This assumes the scaling preserves the aspect ratio, may 
                #     need to store scalex and scaly
                "scale" : token_item.scale() * token_item.pixmap().width(),
                "name":  token_item.childItems()[0].toPlainText()
            }  for token_item in sorted(self.token_items, cmp=lambda a, b: cmp(a.data(0), b.data(0)))
        ]
        d["map_tokens"] = tokens
        pixmaps.update({ token_item.data(0) : token_item.pixmap() for token_item in self.token_items})

        images = [
            {
                "filepath" :  image_item.data(0), 
                "scene_pos" : qtuple(image_item.scenePos()),
                # Note this stores a resolution independent scaling, it has to
                # be divided by the width at load time
                # XXX This assumes the scaling preserves the aspect ratio, may 
                #     need to store scalex and scaly
                "scale" : image_item.scale() * image_item.pixmap().width()
            } for image_item in sorted(self.image_items, cmp=lambda a, b: cmp(a.data(0), b.data(0)))
        ]
        d["map_images"] = images
        pixmaps.update({ image_item.data(0) : image_item.pixmap() for image_item in self.image_items})
        
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
            f.writestr("campaign.json", json.dumps(d, indent=2))
            
            # Zip and store the tokens, images
            # XXX Should probably recreate the fullpaths or create some
            #     uid out of it in case of collision, but would also be 
            #     nice to not duplicate assets shared across several scenes
            #     if a campaign is saved
            #     Use filename + uid?
            
            # XXX Embedding the assets should be optional, eg if it's actively
            #     editing the map in an external tool it's not productive to
            #     have to update the zip file everytime
            # XXX Could do embedded if .qvt, non-embedded if .json
            
            # XXX Have an option to embed the assets as data urls
            
            # For pixmaps, the images may have been downscaled when loaded,
            # store whatever the pixmap contains
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
    
    def populateTokens(self, gscene, scene):
        self.token_items = set()
        for map_token in scene.map_tokens:
            # Note this uses nested items instead of groups since groups change
            # the token rect to include the label but the parent item doesn't
            # contain the child bounding rect
            logger.debug("Populating token %r", map_token.filepath)
            filepath = map_token.filepath
            pix = QPixmap(filepath)
            max_token_size = QSize(64, 64)
            # Big tokens are noticeably slower to render, use a max size
            logger.debug("Loading and resizing token %r from %s to %s", filepath, pix.size(), max_token_size)
            pix = pix.scaled(max_token_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixItem = QGraphicsPixmapItem(pix)
            pixItem.setScale(map_token.scale / pix.width())
            ##pixItem.setGraphicsEffect(QGraphicsOpacityEffect())
            ##pixItem.graphicsEffect().setOpacity(0.5)

            txtItem = QGraphicsTextItem(pixItem)
            # Use HTML since it allows setting the background color
            txtItem.setHtml("<div style='background:rgb(255, 255, 255, 128);'>%s</div>" % map_token.name)
            # Keep the label always at the same size disregarding the token
            # size, because the label is a child of the pixitem it gets affected
            # by it. Also reduce the font a bit
            font = txtItem.font()
            font.setPointSize(txtItem.font().pointSize() *0.75)
            txtItem.setFont(font)
            # XXX This needs to hook on focus out and enter to unfocus and to 
            #     recenter the label and to update the scene tree widget
            #     Note 
            #     txtItem.document().setDefaultTextOption(QTextOption(Qt.AlignCenter))
            #     does center the text in the width set with setTextWidth, but
            #     the item anchor is still the item position so still needs to be
            #     set to the top left corner, which changes depending on the 
            #     text length.
            txtItem.setTextInteractionFlags(Qt.TextEditorInteraction)
            txtItem.setScale(1.0/pixItem.scale())
            # Calculate the position taking into account the text item reverse
            # scale                
            pos = QPointF(
                pixItem.boundingRect().width() / 2.0 - txtItem.boundingRect().width() / (2.0 * pixItem.scale()), 
                pixItem.boundingRect().height() - txtItem.boundingRect().height() / (3.0 * pixItem.scale())
            )
            txtItem.setPos(pos)

            item = pixItem

            item.setPos(*map_token.scene_pos)
            item.setData(0, filepath)
            item.setData(1, map_token)
            item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
            item.setCursor(Qt.SizeAllCursor)

            self.token_items.add(item)

            gscene.addItem(item)
        
    def populateImages(self, gscene, scene):
        self.image_items = set()
        for image in scene.map_images:
            logger.debug("Populating image %r", image.filepath)
            item = QGraphicsPixmapItem(QPixmap(image.filepath))
            item.setPos(*image.scene_pos)
            item.setScale(image.scale / item.pixmap().width())
            item.setData(0, image.filepath)
            ##item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)

            self.image_items.add(item)

            gscene.addItem(item)

    def populateWalls(self, gscene, scene):
        pen = QPen(Qt.cyan)
        self.wall_items = set()
        self.all_walls_item = QGraphicsItemGroup()
        logger.debug("Creating %d wall items", len(scene.map_walls))
        ##scene.map_walls = []
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
        

    def populateDoors(self, gscene, scene):
        pen = QPen(Qt.black)
        brush = QBrush(Qt.red)
        open_brush = QBrush(Qt.green)
        self.door_items = set()
        self.all_doors_item = QGraphicsItemGroup()
        self.open_door_items = set()
        logger.debug("Creating %d door items", len(scene.map_doors))
        for door in scene.map_doors:
            # Doors have been expanded to individual lines for ease of fog
            # calculation, convert to polyline
            lines = door.lines
            points = [QPointF(lines[(i*4) % len(lines)], lines[(i*4+1) % len(lines)]) for i in xrange(len(lines)/4)]
            points.append(QPointF(lines[-2], lines[-1]))
            
            item = QGraphicsPolygonItem(QPolygonF(points))
            item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable)
            item.setPen(pen)
            if (door.open):
                self.open_door_items.add(item)
                item.setBrush(open_brush)
            else:
                item.setBrush(brush)
            item.setData(0, door)
            self.door_items.add(item)
            self.all_doors_item.addToGroup(item)

        gscene.addItem(self.all_doors_item)

    def populateGrid(self, gscene, scene):
        rect = gscene.itemsBoundingRect()
        self.grid_item = QGraphicsItemGroup()
        pen = QPen(QColor(0, 0, 0, 128))
        x, y = rect.left(), rect.top()
        while (x < rect.right()):
            lineItem = QGraphicsLineItem(x, rect.top(), x, rect.bottom())
            lineItem.setPen(pen)
            self.grid_item.addToGroup(lineItem)
            x += scene.cell_diameter

        while (y < rect.bottom()):
            lineItem = QGraphicsLineItem(rect.left(), y, rect.right(), y)
            lineItem.setPen(pen)
            self.grid_item.addToGroup(lineItem)
            y += scene.cell_diameter

        gscene.addItem(self.grid_item)

    def populateGraphicsScene(self, gscene, scene):

        # Populated in z-order
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
                    
            sceneRect = list(self.image_items)[0].sceneBoundingRect()
            f.write('<image href="image.png" x="%f" y="%f" width="%f" height="%f"/>\n' %
                (sceneRect.x(), sceneRect.top(), sceneRect.width(), sceneRect.height()))

            for token_item in self.token_items:
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
        
        if (self.gscene.focusItem() is None):
            logger.warning("Called updateFog with no token item!!")
            return

        map_fog_item = self.map_fog_item
        gscene = self.gscene
        scene = self.scene
        token_pos = self.gscene.focusItem().sceneBoundingRect().center()
        token_x, token_y = qtuple(token_pos)
        logger.info("updateFog %f %f", token_x, token_y)

        logger.info("removing old fog")
        if (map_fog_item is not None):
            gscene.removeItem(map_fog_item)

        fog = None
        fog_polys = []
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

                # Doors are structs, walls are lists
                # XXX Merge walls and convert to structs?
                if (hasattr(wall_or_door, "lines")):
                    wall_or_door = wall_or_door.lines

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

                        fog_polys.append(frustum)
                        
                        item = QGraphicsPolygonItem(QPolygonF([QPointF(p[0], p[1]) for p in frustum]))
                        item.setBrush(brush)
                        item.setPen(pen)
                        fog.addToGroup(item)
                    
            gscene.addItem(fog)

        self.map_fog_item = fog
        self.fog_polys = fog_polys

    def updateImage(self):
        logger.info("updateImage")
        global img_bytes
        gscene = self.gscene
        img_scale = 1.0

        # XXX May want to gscene.clearselection but note that it will need to be
        #     restored afterwards or the focus rectangle will be lost in the DM
        #     view when moving, switching to a different token, etc
        
        # Grow to 64x64 in case there's no scene
        qim = QImage(gscene.sceneRect().size().toSize().expandedTo(QSize(1, 1)) * img_scale, QImage.Format_ARGB32)
        # If there's no current token, there's no fog, just clear to background
        # and don't leak the map
        if (self.gscene.focusItem() not in self.token_items):
            qim.fill(QColor(196, 196, 196))

        else:
            use_svg = False
            if (use_svg):
                self.generateSVG()
                # Use the unfogged image for svg
                # XXX Implement svg fog, image fog or image fog + map
                pix = list(self.image_items)[0].pixmap()
                qim = QImage(pix)
            
            else:
                
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
                p = QPainter(qim)
                gscene.render(p)
                # This is necessary so the painter winds down before the pixmap
                # below, otherwise it crashes with "painter being destroyed
                # while in use"
                p.end()
                
                # Restore all DM user interface helpers
                logger.info("restoring DM ui")
                if (g_draw_walls):
                    self.all_walls_item.show()
                self.all_doors_item.show()

                self.updateFog(g_draw_map_fog, g_blend_map_fog)
                
            
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

        if (self.gscene.focusItem() in self.token_items):
            self.imageWidget.ensureVisible(
                # Zero-based position of the item's center, but in pixmap
                # coordinates
                ((self.gscene.focusItem().sceneBoundingRect().center().x() - self.gscene.sceneRect().x()) * img_scale) * self.imageWidget.scale, 
                ((self.gscene.focusItem().sceneBoundingRect().center().y() - self.gscene.sceneRect().y()) * img_scale) * self.imageWidget.scale,
                self.imageWidget.width() / 4.0, 
                self.imageWidget.height() / 4.0
            )
        
    
    def eventFilter(self, source, event):
        logger.debug("eventFilter source %r type %d", source, event.type())
        
        if ((event.type() == QEvent.GraphicsSceneMouseMove) and 
            (self.gscene.mouseGrabberItem() is not None) and (self.gscene.focusItem() in self.token_items)):
            # XXX This should hook on focus changes too so the viz is updated to
            #     the focused token, but for focus changes mouseGrabberItem is 
            #     (connect to signal focusItemChanged)

            # XXX Create an inherited class and check drag events on items, snap
            #     to grid, draw a grid, etc instead of using the eventfilter

            # Note when the token is a group, the children are the one grabbed,
            # not the group, use the focusitem which is always the group
            # Track the token on the player view
            # No need to update the fog since it will be done when
            # updateImage toggles the fog on and (maybe) off below

            # Update the image if any token (hero or not) moved
            self.updateImage()
            
        elif (event.type() == QEvent.GraphicsSceneMouseDoubleClick):
            if (self.gscene.focusItem() is not None):
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

                elif (focusItem in self.token_items):
                    # XXX Double click to edit size/rotation?
                    logger.info("Editing token %s", focusItem.childItems[0].toPlainText())

            else:
                pen = QPen(Qt.cyan)
                r = self.scene.cell_diameter * 1.0
                sides = 25
                x0, y0 = None, None
                for s in xrange(sides+1):
                    x1, y1 = (event.scenePos().x() + r*math.cos(2.0 * s * math.pi / sides), 
                              event.scenePos().y() + r*math.sin(2.0 * s * math.pi / sides))
                    if (x0 is not None):
                        wall = (x0, y0, x1, y1)
                        self.scene.map_walls.append(wall)

                        item = QGraphicsLineItem(*wall)
                        # Items cannot be selected, moved or focused while
                        # inside a group, the group can be selected and focused
                        # but not moved
                        item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
                        item.setPen(pen)
                        item.setData(0, wall)
                        self.wall_items.add(item)
                        self.all_walls_item.addToGroup(item)
                                            
                    x0, y0 = x1, y1
                    
                self.updateImage()
                self.updateTree()

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

            if ((self.gscene.focusItem() in self.token_items) and (event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down])):
                d = { Qt.Key_Left : (-1, 0), Qt.Key_Right : (1, 0), Qt.Key_Up : (0, -1), Qt.Key_Down : (0, 1)}
                snap_granularity = self.scene.cell_diameter / 2.0
                move_granularity = snap_granularity
                delta = QPointF(*d[event.key()]) * move_granularity


                # snap then move

                # The token item position is offseted so the token appears
                # centered in its position in the map. Take the size of the
                # token into account when calculating the position in the cell
                # XXX This could probably be simplified by using the token's
                # translation matrix?
                snap_pos = ((self.gscene.focusItem().sceneBoundingRect().center() / snap_granularity).toPoint() * snap_granularity)
                
                # Snap in case it wasn't snapped before, this will also allow
                # using the cursor to snap to the current cell if the
                # movement is forbidden below
                
                # Note QSizeF doesn't operate with QPoint, convert to tuple and
                # back to QPoint
                self.gscene.focusItem().setPos(snap_pos-QPointF(*qtuple(self.gscene.focusItem().sceneBoundingRect().size()/2.0)))

                # Intersect the path against the existing walls and doors, abort
                # the movement if it crosses one of those

                # Use a bit of slack to avoid getting stuck in the intersection
                # point due to floating point precision issues, don't use too 
                # much (eg. 1.5 or more) or it will prevent from walking through
                # tight caves
                # XXX Note this doesn't take the size into account, can get
                #     tricky for variable size tokens, since proper intersection
                #     calculation requires capsule-line intersections
                l = QLineF(snap_pos, snap_pos + delta * 1.10)

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
                        # Move set the token corner so the token is centered on
                        # the position in the map
                        self.gscene.focusItem().setPos(snap_pos+delta-QPointF(*qtuple(self.gscene.focusItem().sceneBoundingRect().size()/2.0)))
                    
                self.graphicsView.ensureVisible(self.gscene.focusItem(), self.graphicsView.width()/4.0, self.graphicsView.height()/4.0)
                self.updateImage()
                return True

            elif ((event.key() in [Qt.Key_Tab, Qt.Key_Backtab]) and (event.modifiers() != Qt.ControlModifier)):
                
                if (len(self.token_items) > 0):
                    delta = 1 if (event.key() == Qt.Key_Tab) else -1
                    # XXX Note token_items is a set, so it doesn't preserve the
                    #     order, may not be important since there's no strict
                    #     order between tokens as long as it's consistent one
                    #     (ie the order doesn't change between tab presses as
                    #     long as no items were added to the token set)
                    # XXX Should probably override GraphicsView.focusNextPrevChild
                    #     once moving away from filtering
                    
                    l = list(self.token_items)
                    focused_index = index_of(l, self.gscene.focusItem())
                    if (focused_index == -1):
                        # Focus the first or last item
                        focused_index = len(self.token_items)
                    else:
                        # Clear the selection rectangle on the old focused item
                        l[focused_index].setSelected(False)

                    focused_index = (focused_index + delta) % len(self.token_items)
                    focused_item = l[focused_index]
                    
                    self.gscene.setFocusItem(focused_item)
                    # Select so the dashed rectangle is drawn around
                    focused_item.setSelected(True)
                    self.graphicsView.ensureVisible(focused_item, self.graphicsView.width()/4.0, self.graphicsView.height()/4.0)                    

                    self.updateImage()
                    
                # Swallow the key even if there are no tokens 
                return True
                
            elif ((self.gscene.focusItem() in self.token_items) and (event.text() == " ")):
                # Open the adjacent door
                threshold2 = (self.scene.cell_diameter ** 2.0) * 1.1
                token_center = self.gscene.focusItem().sceneBoundingRect().center()
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

            elif (event.text() == "b"):
                global g_blend_map_fog
                g_blend_map_fog = not g_blend_map_fog

            elif (event.text() == "f"):
                global g_draw_map_fog
                g_draw_map_fog = not g_draw_map_fog

            elif (event.text() == "g"):
                global g_draw_grid 
                g_draw_grid = not g_draw_grid
                self.grid_item.setVisible(g_draw_grid)
                self.graphicsView.update()
            
            elif (event.text() == "m"):
                if (self.player.state() != QMediaPlayer.PlayingState):
                    self.player.play()
                    logger.info("Playing %r", self.playlist.currentMedia().canonicalUrl())
                    
                else:
                    self.player.pause()
                    self.playlist.next()

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