#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

"""
Qt Virtual Table Top
(c) Antonio Tejada 2022

References
- https://github.com/qt/qt5
- https://github.com/qt/qtbase/tree/5.3

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
import StringIO
import sys
import thread
import urllib
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

try:
    import numpy as np
    
except:
    logger.warning("numpy not installed, image filters disabled")
    np = None


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

    # Numpy is only needed to apply edge detection filters for automatic wall 
    # generation
    np_version = "Not installed"
    try:
        import numpy as np
        np_version = np.__version__
        
    except:
        logger.warning("numpy not installed, automatic wall generation disabled")
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
    # XXX This takes multiple seconds eg when "x" cells in the the
    #     combattracker, disabled (probably due to innefficient setscene which
    #     calls addImage, review when scene update is piecemeal)
    return 
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

def os_path_name(path):
    """
    Return name from dir1/dir2/name.ext
    """
    return os.path.splitext(os.path.basename(path))[0]

def os_path_ext(path):
    """
    Return .ext from dir1/dir2/name.ext
    """
    return os.path.splitext(path)[1]

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

def math_norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)

def math_subtract(v, w):
    return (v[0] - w[0], v[1] - w[1])

def math_dotprod(v, w):
    return (v[0] * w[0] + v[1] * w[1])

def bytes_to_data_url(bytes):
    import base64

    base64_image = base64.b64encode(bytes).decode('utf-8')
    # XXX This assumes it's a png (note that at least QTextEdit doesn't care and 
    #     renders jpegs too even if declared as image/png)
    data_url = 'data:image/png;base64,%s' % base64_image

    return data_url


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

    elif (isinstance(q, (QColor))):
        return (q.red(), q.green(), q.blue())

    else:
        assert False, "Unhandled Qt type!!!"

def qlist(q):
    return list(qtuple(q))

alignmentFlagToName = {
    getattr(Qt, name) : name for name in vars(Qt) 
    if isinstance(getattr(Qt, name), Qt.AlignmentFlag)
}
def qAlignmentFlagToString(alignmentFlag):
    return alignmentFlagToName.get(int(alignmentFlag), str(alignmentFlag))

fontWeightToName = {
    getattr(QFont, name) : name for name in vars(QFont) 
    if isinstance(getattr(QFont, name), QFont.Weight)
}
def qFontWeightToString(fontWeight):
    return fontWeightToName.get(int(fontWeight), str(fontWeight))

eventTypeToName = {
    getattr(QEvent, name) : name for name in vars(QEvent) 
    if isinstance(getattr(QEvent, name), QEvent.Type)
}
def qEventTypeToString(eventType):
    return eventTypeToName.get(eventType, str(eventType))

class EventTypeString:
    """
    Using the class instead of the function directly, prevents the conversion to
    string when used as logging parameter, increasing performance.
    """
    def __init__(self, eventType):
        self.eventType = eventType

    def __str__(self):
        return qEventTypeToString(self.eventType)


graphicsItemChangeToName = {
    getattr(QGraphicsItem, name) : name for name in vars(QGraphicsItem) 
    if isinstance(getattr(QGraphicsItem, name), QGraphicsItem.GraphicsItemChange)
}
def qGraphicsItemChangeToString(change):
    return graphicsItemChangeToName.get(change, str(change))

class GraphicsItemChangeString:
    """
    Using the class instead of the function directly, prevents the conversion to
    string when used as logging parameter, increasing performance.

    This has been verified to be a win with high frequency functions like
    itemChange/d
    """
    def __init__(self, change):
        self.change = change

    def __str__(self):
        return qGraphicsItemChangeToString(self.change)

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

def qLaunchWithPreferredAp(filepath):
    # pyqt5 on lxde raspbian fails to invoke xdg-open for unknown reasons and
    # falls back to invoking the web browser instead, use xdg-open explicitly on
    # "xcb" platforms (X11) 
    # See https://github.com/qt/qtbase/blob/067b53864112c084587fa9a507eb4bde3d50a6e1/src/gui/platform/unix/qgenericunixservices.cpp#L129
    if (QApplication.platformName() != "xcb"):
        url = QUrl.fromLocalFile(filepath)
        QDesktopServices.openUrl(url)
        
    else:
        # Note there's no splitCommand in this version of Qt5, build the
        # argument list manually
        QProcess.startDetached("xdg-open", [filepath])

def qImageToDataUrl(imageOrPixmap, imageFormat):
    """
    This works for both QImage and QPixmap because both have the same save
    member function
    """
    ba = QByteArray()
    buff = QBuffer(ba)
    buff.open(QIODevice.WriteOnly) 
    ok = imageOrPixmap.save(buff, imageFormat)
    assert ok
    imgBytes = ba.data()

    dataUrl = bytes_to_data_url(imgBytes)

    return dataUrl

def qKeyPressEventToSequence(event):
    assert event.type() == QEvent.Keypress

    return QKeySequence(event.key() | int(event.modifiers()))

def qEventIsShortcut(event, shortcut_or_list):
    if (isinstance(shortcut_or_list, (list, tuple))):
        return any([qEventIsShortcut(event, shortcut) for shortcut in shortcut_or_list])

    else:
        return (QKeySequence(event.key() | int(event.modifiers())) == QKeySequence(shortcut_or_list))

def qEventIsKeyShortcut(event, shortcut_or_list):
    if (event.type() == QEvent.KeyPress):
        return qEventIsShortcut(event, shortcut_or_list)
    else:
        return False

def qEventIsShortcutOverride(event, shortcut_or_list):
    if (event.type() == QEvent.ShortcutOverride):
        return qEventIsShortcut(event, shortcut_or_list)
    else:
        return False


save_np_debug_images = False
def np_save(a, filepath):
    h, w = a.shape[0:2]
    bits = None
    byte_stride = None
    format = None

    # Convert data type
    if (a.dtype == 'bool'):
        a = a * 255

    elif (a.dtype in ['float32', 'float64']):
        # Assumes range 0 to 1
        a = a * 255.0

    # Note astype copies by default
    a = a.astype(np.uint8)

    # Convert data dimensions
    if ((len(a.shape) == 2) or (a.shape[2] == 1)):
        # Expand one component to three
        # XXX This version of qt has no grayscale8 or grayscale16, replicate to
        #     RGB8
        a = np.repeat(a, 3).reshape((h, w, 3))
        
    assert len(a.shape) == 3
    assert a.shape[2] in [3, 4]
    assert a.dtype == 'uint8'
    
    format = QImage.Format_RGB888 if (a.shape[2] == 3) else QImage.Format_RGBA8888
    image = QImage(a.data, w, h, a.shape[2] * w, format)

    image.save(filepath)

def np_saveset(a, s, value, filepath):
    # Note astype copies by default
    b = a.astype(np.uint8) 
    
    # For some reason b[np.array(list(s)).T] doesn't work (tries
    # to use the arrays as if not transposed) it needs to be
    # copied to intermediate arrays and then used as indices
    j, i = np.array(list(s)).T
    b[j, i] = value
    np_save(b, filepath)

def np_pixmaptoarray(pixmap):
    image = pixmap.toImage()
    pixels = image.constBits().asstring(image.height() * image.bytesPerLine())
    w = image.width()
    h = image.height()
    
    # XXX Needs to check line stride
    # XXX Needs non-alpha support
    a = np.fromstring(pixels, dtype=np.uint8).reshape((h, w, 4)) 

    return a

def np_houghlines(a):
    """
    XXX Not really tested
    """
    def build_hough_space_fom_image(img, shape = (100, 300), val = 1):
        hough_space = np.zeros(shape)
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                if pixel != val : continue
                hough_space = add_to_hough_space_polar((i,j), hough_space)
        return hough_space

    def add_to_hough_space_polar(p, feature_space):
        space = np.linspace(-np.pi/2.0, np.pi/2.0, len(feature_space))
        d_max = len(feature_space[0]) / 2
        for i in range(len(space)):
            theta = space[i]
            d = int(p[0] * np.sin(theta) + p[1] * np.cos(theta)) + d_max
            if (d >= d_max * 2) : continue
            feature_space[i, d] += 1
        return feature_space

    shape = (100, 2*int(math.ceil(np.sqrt(a.shape[0]**2 + a.shape[1]**2))))

    hough = build_hough_space_fom_image(a, shape=shape, val=255)
    # Get sorted indices, increasing values
    idx = np.dstack(np.unravel_index(np.argsort(hough, axis=None), hough.shape))[0]
    lines = []
    for theta, r in idx[-20:]:
        theta = theta * np.pi / hough.shape[0] - np.pi/2.0 
        r = r - hough.shape[1]/2.0
        a = np.cos(theta)
        b = np.sin(theta)
        # x0 stores the value rcos(theta)
        x0 = a*r
        # y0 stores the value rsin(theta)
        y0 = b*r
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + shape[1]*(-b))
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + shape[1]*(a))
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - shape[1]*(-b))
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - shape[1]*(a))

        lines.append((x1, y1, x2, y2))
        
    hough *= 255.0 / hough.max()
    hough = hough.astype(np.uint8)

    if (save_np_debug_images):
        np_save(hough, os.path.join("_out", "hough.png"))

    return lines

def np_filter(a, kernel):
    """
    Convolve the kernel on the array

    The convolution will skip the necessary rows and columns to prevent the
    kernel sampling outside the image.

    @param numeric or boolean 2D array

    @param kernel numeric or boolean kernel, 2D square array

    @return array resulting of convolving the kernel with the array
    """
    logger.info("begin a %s kernel %s", a.shape, kernel.shape)
    # This method 3secs vs. 2mins for the naive method
    # See https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy
    
    # apply kernel to image, return image of the same shape, assume both image
    # and kernel are 2D arrays

    # optionally flip the kernel
    ## kernel = np.flipud(np.fliplr(kernel))  
    width = kernel.shape[0]
    
    # fill the output array with zeros; do not use np.empty()
    b = np.zeros(a.shape)  
    # crop the writes to the valid region (alternatively could pad the source)
    # - For odd kernel sizes, this will skip the same number of leftmost and
    #   rightmost columns and topmost and bottomest rows
    # - For even kernel sizes, this will skip an extra leftmost or topmost
    #   column/row
    bb = b[width/2:b.shape[0]-(width-1)/2,width/2:b.shape[1]-(width-1)/2]
    # shift the image around each pixel, multiply by the corresponding kernel
    # value and accumulate the results
    for j in xrange(width):
        for i in xrange(width):
            bb += a[j:j+a.shape[0]-width+1, i:i+a.shape[1]-width+1] * kernel[j, i]
    # optionally clip values exceeding the limits
    ## np.clip(b, 0, 255, out=b)  

    logger.info("end")
    
    return b

def np_naive_filter(a, kernel):
    logger.info("begin a %s kernel %s", a.shape, kernel.shape)
    kernel_diameter = kernel.shape[0]
    # a = a[::4,::4]
    #a = a[0:a.shape[0]/4, 0:a.shape[1]/4]
    b = np.zeros(a.shape)
    kernel = kernel.reshape(-1)
    sum_kernel = sum(kernel)
    kernel_diameter2 = kernel_diameter / 2

    for j in xrange(0, a.shape[0] - kernel_diameter + 1):
        for i in xrange(0, a.shape[1] - kernel_diameter + 1):
            b[j + kernel_diameter2, i + kernel_diameter2] = (kernel.dot(a[j: j + kernel_diameter, i: i + kernel_diameter].flat) == sum_kernel) * 255

    logger.info("end")
    if (save_np_debug_images):
        np_save(b, os.path.join("_out", "filter.png"))

    return b

def np_erode(a, kernel_width):
    """
    Apply an erosion operator to the 2D boolean array

    @param a boolean 2D array
    
    @param kernel_width width of the erosion kernel to use

    @return boolean 2D array with True on the non eroded elements, False otherwise
    """
    logger.info("a %s kernel %s", a.shape, kernel_width)

    kernel = np.ones((kernel_width, kernel_width))
    kernel_sum = sum(kernel.flat)
    a = np_filter(a, kernel)
    a = np.where(a == kernel_sum, True, False)
    
    if (save_np_debug_images):
        np_save(a, os.path.join("_out", "erode.png"))

    return a

def np_graythreshold(a, threshold):
    # Convert to grayscale
    ##logger.info("Grayscaling")
    a = np.dot(a[...,:4], [0.2989, 0.5870, 0.1140, 0]).astype(np.uint8)
    # Threshold
    ##logger.info("Thresholding")
    # XXX This returns an iterator, not an array, change to np.where?
    a = (a <= threshold)

    if (save_np_debug_images):
        logger.info("Saving threshold")
        np_save(a, os.path.join("_out", "threshold.png"))

    return a

def np_bisect_findmaxfiltersize(a, i, j, threshold):
    """
    Find the largest nxn erode filter size that fits and contains the provided
    array coordinate

    @param a boolean array

    @param x,y coordinates to start probing the filter size

    @return int with max filter size 
    """
    logger.info("i %d j %d", i, j)

    k = 0
    kk = 1
    max_stepsize = 32
    stepsize = max_stepsize
    while (True):
        for jj in xrange(j - kk + 1, j + 1):
            for ii in xrange(i - kk + 1, i + 1):
                # XXX Not clear this just in time thresholding is very
                #     efficient, since it duplicates work already done in the
                #     previous iteration, maybe it should cache some of the
                #     thresholding? (just in time thresholding is known to be 
                #     very beneficial for big maps)
                b = np_graythreshold(a[jj:jj+kk, ii:ii+kk], threshold)
                if (b.all()):
                    k = kk
                    break

            else:
                # Filter didn't fit for any ii, try next jj
                continue
            # Filter did fit, try next size
            break

        else:
            # Filter didn't fit, divide step and try a smaller size
            if ((stepsize == 0) or (kk == 1)):
                # kk will be 1 if even the first iteration fails, abort in that
                # case too
                break
            kk -= stepsize
            stepsize = stepsize / 2
            continue
        # Filter did fit, divide step and try a larger size
        if (stepsize == 0):
            break
        kk += stepsize
        # Can't bisect until the filter fails to fit, otherwise it would only
        # test as far as max_stepsize*2 - 1 (eg for 16, it would test upto
        # 16+8+4+2+1=32-1)
        if (stepsize != max_stepsize):
            stepsize = stepsize / 2

    logger.info("found max %d", k)
    return k

def np_findmaxfiltersize(a, i, j, threshold):
    """
    Find the largest nxn erode filter size that fits and contains the provided
    array coordinate

    @param a boolean array

    @param x,y coordinates to start probing the filter size

    @return int with max filter size 
    """
    logger.info("i %d j %d", i, j)
    
    # XXX Missing padding the original if i, j is too close to the edge or tune
    #     parameters so that part is not probed?
    
    # Find the largest nxn erode filter size that contains the provided array
    # coordinate (so the provided coordinate can be as early as the top left or
    # as late as the bottom right of the nxn erode filter)
    k = 0
    kk = 1
    while (True):
        for jj in xrange(j- kk + 1, j + 1):
            # Test column by column to be able to restart the search skipping
            # columns.
            # XXX This is a dubious optimization, only a perf win vs. the naive
            #     triple loop in very large k sizes like 64, for sizes around 16
            #     the naive and non-naive function time is negligible (less
            #     than the log's default time precision). 
            ii = iii = i - kk + 1
            while (ii < (i + 1)):
                b = np_graythreshold(a[jj:jj+kk,iii], threshold)
                if (b.all()):
                    # This column of the filter fits
                    if (iii == (ii + kk - 1)):
                        # This is the last column of the filter, record the new
                        # max filter size and early exit
                        k = kk
                        break
                    # Continue until the last column of the filter size being
                    # tested is hit
                    iii += 1
                else:
                    # This column doesn't fit at this jj, ii, try the next ii at
                    # this filter size
                    ii += 1
                    iii = ii
            else:
                # No early exit, so filter wasn't found at this jj, try next jj
                continue
            # Early exit, so this filter size was found to fit, done testing
            # this j,i at this filter size
            break
        else:
            # The loop didn't early exit, this means a kk x kk containing i,j
            # coudln't fit, no larger filter sizes containing i,j will fit
            # either, done
            break
        # Try next filter size
        kk += 1
    
    logger.info("found max k %d", k)

    return k

def np_floodfillerode(a, x, y, erosion_diameter):
    """
    @param a Boolean ndarray

    @param x,y position inside the ndarray to start flooding

    @param erosion_diameter, flood only if all the pixels in the
           erosion_diameter / 2, erosion_diameter/2 + 1 are true

    @return Python set of j, i tuples
    """
    logger.info("x %d y %d", x, y)

    # XXX This needs to add padding
    stack = []
    if (a[y, x]):
        stack.append((y, x))
    active = set(stack)

    # Flood fill into the active set
    logger.info("Flooding")
    while (len(stack) > 0):
        pos = stack.pop()
        # Check the 8 directions
        # XXX This needs to check there's enough padding for a 3x3 matrix
        for delta in [(0,1), (1,0), (0,-1), (-1,0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            t = (pos[0] + delta[0], pos[1] + delta[1])
            # Do just in time eroding
            # XXX Do just in time scaling too? but just in time
            #     downscaling complicates things down the line with
            #     the deltas below? (would be like a sparse flood?)
            # XXX Do just in time grayscale and thresholding, may not be that
            #     beneficial because filter stamps overlap and they will be
            #     applied multiple times for the overlapped regions and those
            #     are very fast whole image operations
            if ((t not in active) and
                # XXX Do bounds checking by having a guardband?
                ((erosion_diameter / 2) < t[0] < (a.shape[0] - (erosion_diameter / 2))) and
                ((erosion_diameter / 2) < t[1] < (a.shape[1] - (erosion_diameter / 2))) and
                (a[
                    t[0] - erosion_diameter / 2 : t[0] + erosion_diameter / 2 + 1, 
                    t[1] - erosion_diameter / 2 : t[1] + erosion_diameter / 2 + 1
                ].all())):
                    active.add(t)
                    stack.append(t) 
    
    if (save_np_debug_images):
        logger.info("Saving flood")
        np_saveset(a, active, 128, os.path.join("_out", "flood.png"))

    return active

def np_findcontourpoints(a, active):
    """
    Thin contour using "A fast parallel algorithm for thinning digital patterns"
    by Zhang Suen '84

    @param a boolean ndarray

    @param active Set of connected coordinates to simplify the contour for

    @returm thinned contour as a set of (j,i) tuples
    """
    changes = True
    even = True
    iteration = 0
    cache = dict()
    cache_misses = 0
    cache_queries = 0
    while (changes or not even):
        logger.info("Contouring %d, %d points" % (iteration, len(active)))
        erased = set()
        changes = False
        for pos in active:
            # Use a cache to store the results, this ends up having a very high
            # hit ratio > 99%

            # XXX Could keep the cache across invocations since it's small
            #     (10-item boolean keys, so 1024 entries) and data-independent
            # XXX Could also prebuild the cache offline or at startup
            key = tuple([(pos[0] + (ji / 3) - 1, pos[1] + (ji % 3) - 1) in active for ji in xrange(9)] + [even])
            erase = cache.get(key, None)
            cache_queries += 1
            if (erase is None):
                cache_misses += 1
                # Build neighbors array
                p = []
                psum = 0
                dprev = None
                count01 = 0
                # XXX This needs to check there's enough padding for a 3x3
                #     matrix
                for delta2 in [
                    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
                    ]:
                    t = (pos[0] + delta2[0], pos[1] + delta2[1])
                    d = int(t in active)
                    p.append(d)
                    if ((dprev == 0) and (d == 1)):
                        count01 += 1
                    dprev = d
                    psum += d
                if ((dprev == 0) and (p[0] == 1)):
                    count01 += 1
                    
                # Mark for deletion if
                erase = (
                    # 2 <= B(Pi) <= 6, B(Pi) = sum(P2-9)
                    (2 <= psum <= 6) and
                    # A(P1) = 1, number of 01 patterns in P2-9
                    (count01 == 1) and
                    (
                        (even and (
                            # Remove south or east boundary and north-west corner
                            # P2 * P4 * P6 = 0
                            ((p[2 - 2] * p[4 - 2] * p[6 - 2]) == 0) and
                            # P4 * P6 * P8 = 0
                            ((p[4 - 2] * p[6 - 2] * p[8 - 2]) == 0)
                        )) or 
                        ((not even) and (
                            # Remove north or west boundary or south-east corner
                            # P2 * P4 * P8 = 0
                            ((p[2 - 2] * p[4 - 2] * p[8 - 2]) == 0) and
                            # P2 * P6 * P8 = 0
                            ((p[2 - 2] * p[6 - 2] * p[8 - 2]) == 0)
                        )
                    ))
                )
                cache[key] = erase

            if (erase):
                erased.add(pos)
                changes = True
                
        active = active - erased
        even = not even
        if (save_np_debug_images):
            logger.info("Saving %d points in contour %d" % (len(active), iteration))
            np_saveset(a, active, 128, os.path.join("_out", "contour%d.png" % iteration))
        iteration += 1

    logger.info("Contour cache %0.2f%% miss" % (cache_misses * 100.0 / cache_queries))
    if (save_np_debug_images):
        logger.info("Saving contour")
        np_saveset(a, active, 128, os.path.join("_out", "contour.png"))

    return active

def np_connectcontourpoints(a, active):
    """
    Connect the points in active set in polylines of lines 1-pixel long.

    The active set is supposed to be a single-pixel-width contour (with or
    without stair-steps, open or closed)

    @param a bool ndarray 

    @param active set of positions 

    @return list of polylines (list of list of positions) and set of tjunctions
            (set of positions)
    """
    # Collect all the points in the active set into sequential connected points

    # XXX This should also find loops, probably by replicating the last point so
    #     it can be easily detected later?
    logger.info("Lining %d points", len(active))
    polylines = []
    pos = None
    processed = set()
    tjunctions = set()
    while (len(active) > 0):
        if (pos is None):
            pos = active.pop()
            processed.add(pos)
            start = pos
            polyline = [pos]
            polylines.append(polyline)

        # This purporsefully searches for horizontal and vertical connections
        # (stair step lines) before diagonal connections (diagonal lines) so
        # stair-steps are picked before diagonal connections, otherwise isolated
        # segments walls would appear since the edge thinning generates
        # stair-steps sometimes instead of diagonals
        for delta in [(0,1), (1,0), (0,-1), (-1,0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            t = (pos[0] + delta[0], pos[1] + delta[1])
            if (t in active):
                processed.add(t)
                active.remove(t)
                polyline.append(t)
                pos = t
                break

        else:
            # The line hit an extreme, either change direction or finish this
            # polyline and start a new one

            # The conversion into polylines, breaks t-junctions (points that are
            # shared by multiple lines) causing discontinuities. If the polyline
            # ends in an already-processed point (ie it's a t-junction), link it
            # to that point so the discontinuity visually disappears
            pos = polyline[-1]
            for delta in [(0,1), (1,0), (0,-1), (-1,0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                t = (pos[0] + delta[0], pos[1] + delta[1])
                # Note polyline can have only one vertex if the
                # other one is a t-junction
                if ((t in processed) and ((len(polyline) == 1) or (t != polyline[-2]))):
                    polyline.append(t)
                    tjunctions.add(t)
                    break

            # Walk some other non-starting direction in case the starting point
            # was in the middle of the line. Note there's no need to store the
            # starting direction since the neighbors in the starting direction
            # have been removed (there will be a few wasted delta tests but
            # that's all)
            # XXX In case of three or more directions it could try to join the
            #     longest segments
            if (start is not None):
                polyline.reverse()
                pos = polyline[-1]
                start = None

            else:
                pos = None
    
    return polylines, tjunctions

def np_coalescepoints(a, polylines, tjunctions, max_coalesce_dist2):
    """
    Simplify the polylines

    @param a bool ndarray

    @param polylines

    @param tjunctions
    
    @return Simplified polylines
    """
    def add_to_hough_space(p, theta_space, feature_space, d_max):
        d_max2 = d_max * 2
        # Note these are vector operations
        d = (p[0] * sin + p[1] * cos).astype(int)
        d = ((d + d_max) * len(feature_space[0])) / d_max2
        feature_space[:, d] += 1

        return feature_space

    logger.info("Coalescing %d points %d tjunctions", sum([len(polyline) for polyline in polylines]), len(tjunctions)) 
    
    # Coalesce colinear points
    d_max = int(math.ceil(np.sqrt(a.shape[0]**2 + a.shape[1]**2)))
    # Quantize the space for better coalescing, the specific values have been
    # picked empirically from visual inspection
    # XXX Investigate why quantizing theta doesn't see to work as well as
    #     quantizing rho
    shape = (17, (2 * d_max) / 3)
    theta_space = np.linspace(-np.pi/2.0, np.pi/2.0, shape[0])
    sin = np.sin(theta_space) 
    cos = np.cos(theta_space)
    # Setting this larger than 1 makes corners not accurate enough, so when that
    # happens in a vertical corner, then the horizontal line will miss the real
    # wall and it doesn't seem to help enough removing wiggle points from
    # straight lines
    max_line_length_diff = 1
    
    # Coalesce points that are too close to each other
    # XXX Coalescing points by proximity is not really well tested with colinear
    #     removal, seems to remove too many points as is, needs investigating
    coalesce_walls = False
    coalesced_walls = 0

    walls = []
    for polyline_index, polyline in enumerate(polylines):
        logger.info("Coalescing polyline %d/%d", polyline_index + 1, len(polylines))
        prevpos = polyline[0]
        vstart = [0, 0]
        start = prevpos
        points = []
        hough_space = np.zeros(shape)
        # This will force a flush of the line in the first iteration
        # XXX Unless distance coalescing is enabled?
        hough_line_length = max_line_length_diff + 1
        for pos in polyline[1:]:
            x1, y1 = pos
            x0, y0 = prevpos

            if (coalesce_walls and (prevpos not in tjunctions) and (
                # This edge is small
                ((x1 - x0) ** 2 + (y1 - y0) ** 2 < max_coalesce_dist2) and
                # The distance between the start and the end is less than min
                (((x1 - start[1]) ** 2) + ((y1 - start[0]) ** 2) < max_coalesce_dist2)
            )):
                logger.debug("Coalescing %s to %s", (x1, y1), (start[1], start[0]))
                coalesced_walls += 1
                prevpos = pos
                continue

            start = (y1, x1)

            # Force a flush on non-colinear or t-junction (avoid removing
            # t-junctions since it causes discontinuities)
            # XXX Could remove a t-junction if all but one joined lines are
            #     removed?

            # Do colinear tests using a hough space, note that the usual method
            # of dot product is not that useful because the delta between
            # consecutive positions is -1, 1 or 0, so there are only 9 possible
            # values
            hough_space = add_to_hough_space(pos, theta_space, hough_space, d_max)
            hough_line_length += 1
            
            if (((hough_line_length - hough_space.max()) > max_line_length_diff) or (prevpos in tjunctions)):
                hough_space = np.zeros(shape)
                hough_space = add_to_hough_space(pos, theta_space, hough_space, d_max)
                hough_line_length = 1
                points.append([prevpos[1], prevpos[0]])
                ##logger.info("reset hough with pos %s length %d max %d", pos, hough_line_length, hough_space.max())

            else:
                pass
                ##logger.info("added pos %s length %d max %d", pos, hough_line_length, hough_space.max())
                
            prevpos = pos
        if ((len(points) <= 1) or (pos != (points[-1][1], points[-1][0]))):
            points.append([pos[1], pos[0]])
        walls.append(points)

    logger.info("Walling %d walls, %d points, %d coalesced", len(walls), 
        sum([len(wall) for wall in walls]), coalesced_walls)

    return walls


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

def import_ds_walls(scene, filepath):
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
    coalesced_walls = 0
    coalesce_walls = True
    coalesce_colinear = True
    colinear_walls = 0
    colinear_threshold = 0.2
    for layer in layers:
        shape1_is_polyline = False
        if (ds_version == 1):
            # Doors on v1 are three squares (so 24 points) but also lots of 
            # replicated points at the intersections:
            # - 10 left side between top and mid 
            # - 10 left side at top
            # - ...
            # all in all, 45 points followed by 25 followed by 25
            # 25 are the door hinges, 45 is the door pane
            
            # XXX Open doors on v1 don't let line of sight go through, looks
            #     like there's extra geometry that prevents it, investigate (or
            #     remove manually once map is editable). On simplev1.qvt these
            #     turn out to be wall indices 4-5 and 16-17 on a 25-point wall
            #     resulting of coalescing a 153-point shape2 so it doesn't look
            #     like it comes from hinges (which are coalesced from 25 to 4
            #     points). Splitting on those wall indices into polylines fixes
            #     the problem but it doesn't look like it will scale to other
            #     maps. The only solution seems to be manually edit the walls.
            is_door = False
            shapes = layer.get("shape", {}).get("shapeMemory", [[]])
            # Looks like the shape memory is some kind of snapshots/undo
            # history, only look at the latest version
            currentShapeIndex = layer.get("shape", {}).get("currentShapeIndex", 0)
            shapes = shapes[currentShapeIndex]

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
            # Polygons can be walls or door panes, polygons replicate the first
            # point in the last but door panes don't
            shapes = layer.get("polygons", [])
            # Polylines are empty on v1 and correspond to door hinges and
            # polyline walls on v2, adding as walls
            polylines = layer.get("polylines", [])

            if (len(polylines) > 0):
                # Create a fake shape so the polylines go through all the
                # coalescing, etc of regular shapes
                shapes.insert(0, polylines)
                shape1_is_polyline = True

        for shape1_index, shape1 in enumerate(shapes):
            shape1_is_door = is_door and (not shape1_is_polyline)
            
            for shape2 in shape1:
                points = []
                x0, y0 = None, None
                vstart = [0, 0]
                colinear_length = 0
                
                if (ds_version == 1):
                    # 25 is the door hinges, 45 is the door pane
                    shape1_is_door = (len(shape2) in [45])
                    
                for pos in shape2:
                    x1, y1 = pos

                    if (x0 is not None):
                        vpos = math_subtract(pos, prevpos)
                        norm = math_norm(vpos)
                        vpos = (0, 0) if (norm == 0) else (vpos[0] / norm, vpos[1] / norm)
                        
                        # XXX There are also line overlaps, and polylines going
                        #     back and forth, but overlap removal probably needs
                        #     to be done at global level and as a preprocess?
                            
                        # Remove unnecessary middle points on straight lines, v2
                        # is known to generate these on polylines
                        if (coalesce_colinear and 
                            # Note -1 dotprod means opposite directions,
                            # coalesce only positive 1 as negative means the
                            # polyline going back and forth
                            (math_dotprod(vpos, vstart) > (1.0 - colinear_threshold))):
                            # XXX Don't remove t-junctions, since there's some
                            #     threshold it will cause gaps?
                            logger.debug("Colinear coalescing %s to %s length %d", (x1, y1), start, colinear_length)
                            
                            colinear_walls += 1
                            colinear_length += 1
                        
                        else:
                            colinear_length = 0
                        
                        # Coalesce walls that are too small, this fixes
                        # performance issues when walls come from free form wall
                        # tool with lots of very small segments

                        # Don't coalesce doors as they may have small features
                        # on v1
                        if (coalesce_walls and (not shape1_is_door) and (
                            # This edge is small
                            ((x1 - x0) ** 2 + (y1 - y0) ** 2 < max_coalesce_dist2) and
                            # The distance between the start and the end is less than min
                            (((x1 - start[0]) ** 2) + ((y1 - start[1]) ** 2) < max_coalesce_dist2)
                        )):
                            logger.debug("Point coalescing %s to %s", (x1, y1), start)
                            coalesced_walls += 1

                        else:
                            x0, y0 = start
                            # Doors and polylines need the last point, polygons
                            # replicate the first point in the last point, so
                            # skip that one for polygons
                            if (shape1_is_polyline or shape1_is_door or (pos is not shape2[-1])):
                                # Keep overwriting colinear points, otherwise
                                # append
                                if (colinear_length >= 1):
                                    points[-1] = [x1, y1]

                                else:
                                    points.append([x1, y1])
                            
                            start = x1, y1
                            vstart = vpos

                    else:
                        start = x1, y1
                        points.append([x1, y1])
            
                    x0, y0 = x1, y1
                    prevpos = pos

                # Remove walls/polylines/doors reduced to 0 or 1 points (due to
                # coalescing or otherwise)
                if (len(points) > 1):
                    if (shape1_is_door):
                        map_doors.append(Struct(points=points, open=False))

                    else:
                        map_walls.append(Struct(points=points, closed=not shape1_is_polyline))
            shape1_is_polyline = False

    logger.info("Found %d doors %d valid walls, %d coalesced, %d colinear", len(map_doors), 
        len(map_walls), coalesced_walls, colinear_walls)
    
    scene.cell_diameter = map_cell_diameter
    scene.map_walls = map_walls
    scene.map_doors = map_doors


def import_ds(scene, ds_filepath, map_filepath=None, img_offset_in_cells=None):
    import_ds_walls(scene, ds_filepath)

    if (map_filepath is not None):

        # Size in cells is embedded in the filename 
        m = re.match(r".*\D+(\d+)x(\d+)\D+", map_filepath)
        img_size_in_cells = (int(m.group(1)), int(m.group(2)))
        logger.debug("img size in cells %s", img_size_in_cells)

        if (img_offset_in_cells is None):
            # Make a best guess at the alignment matching the center of the wall 
            # bounds with the center of the grids
            walls = []
            for wall in scene.map_walls:
                walls.extend(wall.points)
            bounds = (
                min([x for x, y in walls]), min([y for x, y in walls]),
                max([x for x, y in walls]), max([y for x, y in walls])
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

def build_index(dirpath):
    logger.info("%r", dirpath)
    words = dict()
    filepath_to_title = dict()
    title_to_filepath = dict()
    
    # XXX webhelp\mm has forced brs which look double spaced on QTextEdit,
    #     reformat and fix so everything comes from webhelp? (look ok-ish on
    #     Edge but not on QTextBrowser). Also has forced font and size HTML
    #     attribs
    # XXX Get all these from the public Fantasy Grounds 2E pack instead?
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
        logger.info("source %s type %s", class_name(source), EventTypeString(event.type()))
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
        self.displayInitiativeOrder = True

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
        button = QPushButton("Clear Rolls")
        self.rollAttackButton = button
        button.clicked.connect(self.clearRolls)
        hbox.addWidget(button)
        hbox.setStretchFactor(button, 0)
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
            with (QSignalBlocker(self.table)):
                self.sceneChanged.emit()

    def clearRolls(self):
        logger.info("")
        table = self.table

        # XXX Each cell modification causes a scene update, block updates and
        #     unblock at the end. Remove once scene updates are piecewise
        self.setUpdatesBlocked(True)

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        for j in xrange(table.rowCount()):
            # Don't clear those tagged as PC, also don't clear those that are 
            # already empty since they could be in use in a different combat 
            # tracker (but there's no need to check for that since setting the
            # empty value again won't trigger a change and won't mess with
            # the tokens owned by that other combat tracker)
            # XXX Eventually combat trackers will only contain the tokens they
            #     own instead of all the tokens in the scene and and the above
            #     comment can be tweaked
            if (table.item(j, self.headerToColumn["PC"]).text() == ""):
                table.item(j, self.headerToColumn["I"]).setText("")
                table.item(j, self.headerToColumn["A(D)"]).setText("")
        
        table.setSortingEnabled(True)

        self.setUpdatesBlocked(False)


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

    def setDisplayInitiativeOrder(self, display):
        # XXX This and the other combattracker updates resets the player view,
        #     find out why
        logger.info("display %s", display)
        table = self.table

        # XXX Each cell modification causes a scene update, block updates and
        #     unblock at the end. Remove once scene updates are piecewise
        self.setUpdatesBlocked(True)

        # Disable sorting otherwise items will sort themselves mid-insertion
        # interfering with insertion
        table.setSortingEnabled(False)

        self.displayInitiativeOrder = display
        for j in xrange(table.rowCount()):
            # Only modify the tockens tracked in this combat tracker (PC or not,
            # but only with non empty initiative)
            item = table.item(j, self.headerToColumn["I"])
            if (item.text() != ""):
                if (display):
                    # Toggle the value to trigger writing the right value to center
                    # (setting the same value doesn't trigger, needs to be toggled)
                    text = item.text()
                    item.setText("")
                    item.setText(text)

                else:
                    # Clear the center label, this combattracker is no longer 
                    # using it
                    item.data(Qt.UserRole).center = ""

        table.setSortingEnabled(True)

        self.setUpdatesBlocked(False)

    def rollInitiative(self):
        logger.info("")
        table = self.table

        # XXX If multiple cells or a row selected, roll only for that row

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

        table.setSortingEnabled(True)

        self.setUpdatesBlocked(False)

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

        elif ((header == "I") and (self.displayInitiativeOrder)):
            # XXX This assumes multiple combattrackers target disjoint tokens
            if (text == ""):
                token.center = ""

            else:
                # Find out the initiative order 
                items = [table.item(i, col) for i in xrange(table.rowCount()) ]
                # Sort, put empties at the end
                def compInitiative(a, b):
                    if (a.text() == ""):
                        return cmp(1, 0)
                        
                    elif (b.text() == ""):
                        return cmp(0, 1)
                    
                    else:
                        # XXX This should look at something to break ties? in theory
                        #     the tie order should be consistent as long as there
                        #     are no insertions in the table
                        return cmp(float(a.text()), float(b.text()))

                # XXX Cache this list?

                # XXX This has to match the order when sorting by initiative
                #     column, which seems to be the case (other than sorting
                #     empties at the top vs. bottom), but change the column sort
                #     so it uses the same comparison function to be sure?
                # XXX This should probably be saved in the ruleset data in case
                #     encounters are resumed across sessions?
                items = sorted(items, cmp=compInitiative)

                # By changing the initiative of this token, the initiative order
                # of all tokens may have changed, update all tokens
                for i, item in enumerate(items):
                    if (item.text() != ""):
                        item.data(Qt.UserRole).center = "%d" % (i+1)
            
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

        with QSignalBlocker(table):
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
                    logger.debug("setting data %d, %d, %s", row, i, token.name)
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
        logger.debug("source %s type %s", class_name(source), EventTypeString(event.type()))
        if ((source == self.encounterTable) and qEventIsShortcutOverride(event, "del")):
            logger.info("ShortcutOverride for %d", event.key())
            # Ignore the global "del" key shortcut so cells can be edited,
            # XXX Fix in some other way (per dockwidget actions?) and remove,
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

class VTTTextEditor(QTextEdit):
    """
    Text editor with markdown-feature level support:
    - headings
    - bullet/numbered lists
    - blockquote
    - paragraph alignment (left, justified, center, right)
    - text formatting (bold/italic/underline)
    - tables
    - images
    
    References
    - https://doc.qt.io/qt-5/richtext-html-subset.html
    - https://doc.qt.io/qt-5/richtext-structure.html
    - https://doc.qt.io/qt-5/richtext.html
    - https://github.com/mfessenden/SceneGraph/blob/master/qss/stylesheet.qss
    - https://github.com/qt/qtbase/blob/5.3/src/gui/text/qtextdocument.cpp

    Implementation 
        This uses Qt api instead of inserting HTML text because 
        - <span> and <p> can be inserted in the html, but they are not preserved
        - QT removes HTML tags with no content, so there's no way to set the
          format of an empty paragraph
    """
    # XXX Format as link with ctrl+k? (setAnchor, setAnchorHref, etc), set
    #     LinksAccessibleByMouse 
    formatShortcuts = ["ctrl+1", "ctrl+2", "ctrl+3", "ctrl+4", "ctrl+b", "ctrl+i", "ctrl+l", "ctrl+q", "ctrl+t", "ctrl+u"]
    tokenToListStyle = { 
        "-" : QTextListFormat.ListSquare, 
        "*" : QTextListFormat.ListDisc, "+" : QTextListFormat.ListCircle, 
        "1." : QTextListFormat.ListDecimal,
        "A." : QTextListFormat.ListUpperAlpha, "a." : QTextListFormat.ListLowerAlpha, 
        "I." : QTextListFormat.ListUpperRoman, "i." : QTextListFormat.ListLowerRoman, 
    }
    
    def __init__(self, *args, **kwargs):
        super(VTTTextEditor, self).__init__(*args, **kwargs)

        # XXX Use some style sheet instead of hard-coding here?
        fontId = QFontDatabase.addApplicationFont(os.path.join("_out", "fonts", "Montserrat-Light.ttf"))
        logger.info("Font Families %s",QFontDatabase.applicationFontFamilies(fontId))
        # XXX Looks like when exporting to HTML there are spurious quote marks,
        #     probably because of the whitespace?
        font = QFont("Montserrat Light")
        font.setPointSize(10)
        self.setFont(font)
        self.setTextInteractionFlags(self.textInteractionFlags() | Qt.LinksAccessibleByMouse)

        self.document().contentsChange.connect(self.contentsChange)
        self.installEventFilter(self)
        
        # XXX Get Trilium styles from 
        #     https://github.com/zadam/trilium/blob/master/src/public/stylesheets/style.css
        # See https://doc.qt.io/qt-5/richtext-html-subset.html
        # XXX Find a way of doing single border tables, no style seems to work
        #     unless put directly as attribute in the table or using the Qt API?
        self.document().setDefaultStyleSheet("""
            table {
                /* border-collapse: collapse; */ /* Doesn't work to get single border */
                /* border-top-style: none; */
                border-width : 1;  /* works */
                /* border-top-width : 1px; */
                /* border-style: dashed; */ /* Works */
                border-style: ridge; /* Doesn't work */
                /* border-color: red;*/ /* Works */
                /* border-bottom-color: red; */ /* Doesn't work */
                /* border settings (top-width, style, etc don't seem to work */
                /* border-spacing: 0; */ /* Doesn't work, cellspacing works as table attribute, but not as CSS */
            }
            th { 
                background-color: lightgrey;
                color: black; 
            }
            /* tr { 
                border-width: 0px;
                border-style: solid;
            }
            td { 
                border-width: 0px;
                border-style: solid;
            }*/
        """)

    def paintEvent(self, event):
        """
        Paint a vertical line on the left margin of indented blocks
        (blockquotes)
        """
        super(VTTTextEditor, self).paintEvent(event)
        
        # Find indentation blocks and draw a vertical line on the left margin
        block = self.document().begin()
        while (block.isValid()):
            # XXX Could also use setUserData or setUserState to detect them, (or
            #     store them in some set/list, but keeping it uptodate can be an
            #     issue). Currently this also detects list blocks where the
            #     bullet has been removed via backspace
            if (block.blockFormat().indent() > 0):
                cursor = QTextCursor(block)
                blockLayout = block.layout()
                rect = blockLayout.boundingRect()
                rect.translate(blockLayout.position())
                painter = QPainter(self.viewport())
                # The viewport painter is not translated to the scroll position
                # by default, needs explicit translation
                painter.translate(-self.horizontalScrollBar().value(), -self.verticalScrollBar().value())
                painter.fillRect(
                    rect.x() - self.document().indentWidth() / 2, 
                    rect.y(), 
                    5, rect.height(), 
                    Qt.gray)
                # Not using .end() causes a silent app crash
                painter.end()
                
            block = block.next()

    def eventFilter(self, source, event):
        logger.debug("source %s type %s", class_name(source), EventTypeString(event.type()))
        if ((source == self) and qEventIsShortcutOverride(event, VTTTextEditor.formatShortcuts)):
            logger.info("Control char %r %d", event.text(), event.key())
            # Ignore some global keys so they can be used for formatting text
            # XXX Fix in some other way (per dockwidget actions?) and remove,
            #     see Qt.ShortcutContext
            # Note even with shortcutoverride widget-local QActions won't
            # work
            logger.info("ShortcutOverride for %d", event.key())
            event.accept()

            return True

        elif ((source == self) and qEventIsKeyShortcut(event, VTTTextEditor.formatShortcuts)):
            self.toggleFormat(QKeySequence(event.key()).toString().lower())
            
            return True
        
        return False

    def toggleFormat(self, formatChar):
        """
        If some text is selected, format as given, otherwise change current
        format. If the text or current format already have that format, remove
        that format

        formatChar is one of 
        - "b", "i", "u" (bold, italic, underline)
        - "t" (left, center, right, justified alignment)
        - "1", "2", "3", "4" (Heading level, level 1 is largest font)
        - "q" toggle blockquote
        - "l", "ll", "lll", "llll" create/cycle list style with as many "l" as
          index to styles + 1 (circle, square, disc, numeric, non-numeric..)
        """
        # XXX This should allow for reverse toggle in case of multiple toggles
        #     (eg back cycling through list styles), probably using upper vs.
        #     lower case format char? (testing for shift modifier here is not
        #     clean since it's called from eg token parsing)

        logger.info("formatChar %r", formatChar)
        # XXX Every setTextCursor causes a textChanged, should block signals
        #     until the last one?
        # XXX This should use beginEditBlock so the operations are atomic from
        #     undo/redo point of view?
        # Note this works for setting both the selected text format or the
        # current format if no selection
        prevCursor = self.textCursor()
        cursor = self.textCursor()
            
        if (formatChar in ["1", "2", "3", "4"]):
            # Block format
            
            # Note blockCharFormat is not the current block format but the
            # format when inserting at the beginning of an empty block. 
            # Select and change the charformat from the start to the end
            # of the block

            # Note StartOfLine is the start of the wrapped line, not the
            # physical line, use StartOfBlock instead
            # XXX This doesn't work when selecting multiple paragraphs, would 
            #     need to iterate since it needs to add/remove rulers?
            cursor.movePosition(QTextCursor.StartOfBlock)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            fmt = cursor.charFormat()
            # XXX This should remove all other formats?

        elif (formatChar == "t"):
            # Cycle through paragraph text alignments
            # XXX This currently affects headings too, prevent?
            # XXX Make this affect the table position (works for cells but not
            #     for centering the table in the page)
            # XXX Needs image sizing options (width, 1/2 size, etc) or do it at
            #     import/paste time? Note the image sizing option needs to be 
            #     HTML compatible so they can be saved and restored
            alignments = [Qt.AlignLeft, Qt.AlignJustify, Qt.AlignHCenter, Qt.AlignRight]
            alignment = self.alignment()
            i = index_of(alignments, alignment)
            delta = -1 if (int(qApp.keyboardModifiers() & Qt.ShiftModifier) != 0) else 1
            alignment = alignments[(i + delta) % len(alignments)]
            self.setAlignment(alignment)

        elif (formatChar.startswith("l")):
            styles = sorted(VTTTextEditor.tokenToListStyle.values())
            textList = cursor.currentList()
            i = formatChar.count("l") - 2
            
            if (textList is not None):
                # Modify the current list style
                if (i == -1):
                    # No style provided, cycle through styles
                    # XXX Remove list on wrap around?
                    i = index_of(styles, textList.format().style())
                    i = (i + 1) % len(styles)

                fmt = textList.format()
                fmt.setStyle(styles[i])
                fmt = textList.setFormat(fmt)

            else:
                # Create a list with the cursor as first item and the given
                # style
                i = max(0, i)
                cursor.createList(styles[i])

        elif (formatChar == "q"):
            # Block-align the selection in case text from multiple blocks is
            # selected
            start = QTextCursor(prevCursor)
            start.setPosition(prevCursor.selectionStart())
            start.movePosition(QTextCursor.StartOfBlock)

            end = QTextCursor(prevCursor)
            end.setPosition(prevCursor.selectionEnd())
            end.movePosition(QTextCursor.EndOfBlock)

            cursor.setPosition(start.position())
            cursor.setPosition(end.position(), QTextCursor.KeepAnchor)
            
            indented = (cursor.blockFormat().indent() != 0)

            # Toggle indent
            fmt = QTextBlockFormat()
            # Set the indent via setIndent, another option is to set the
            # leftMargin (eg fmt.setLeftMargin(30))
            # - setLeftMargin appears on the HTML but it's fixed and cannot be
            #   removed with the backspace key
            # - setIndent doesn't show in the HTML (uses -qt-text-indent private
            #   attribute) but it's relative to document().indentWidth and can
            #   be removed with the backspace key
            fmt.setIndent(0 if indented else 1)
            cursor.mergeBlockFormat(fmt)
            
            # Toggle italic
            # Note fmt is used in the generic charformat block later on in case
            # there's no selection
            fmt = QTextCharFormat()
            fmt.setFontItalic(not indented)
            cursor.mergeCharFormat(fmt)

            self.setTextCursor(cursor)

            # Put the cursor back to where it was
            self.setTextCursor(prevCursor)

            

        else:
            # Char format

            # Always look at the format at the end of the selection in case the
            # selection is reversed and the cursor points to the char *before*
            # the selection. This also works if there's no selection since
            # selectionEnd() returns position() if no selection
            c = self.textCursor()
            c.setPosition(c.selectionEnd())
            fmt = c.charFormat()

        if (formatChar == "b"):
            fmt.setFontWeight(QFont.Normal if (fmt.fontWeight() == QFont.Bold) else QFont.Bold)

        elif (formatChar == "i"):
            fmt.setFontItalic(not fmt.fontItalic())

        elif (formatChar == "u"):
            fmt.setFontUnderline(not fmt.fontUnderline())

        elif (formatChar in ["1", "2", "3", "4"]):
            # XXX If heading is used on a ruler block it should move up or
            #     ignore? But note that the ruler can be moved manually so the
            #     block above may not be the heading so it should just ignore?
            #     (but it should also ignore on tables, images, etc?)
            level = int(formatChar)
            hLevel = fmt.property(QTextFormat.FontSizeAdjustment)
            if (hLevel is not None):
                hLevel = 3 - hLevel + 1
            # Note this could insertHtml with the appropriate heading, but 
            # html formatting is removed by QtTextEditor if there are no chars
            # so format cannot be set on an empty paragraph, use FontSizeAdjustment
            # property instead, see
            # https://github.com/qt/qtbase/blob/5.3/src/gui/text/qtextdocument.cpp#L2446
            # https://github.com/qt/qtbase/blob/5.14/src/gui/text/qtextmarkdownimporter.cpp
            # 0: medium, 1: large, 2: x-large, 3: xx-large, 4: xxx-large
            # Note in the sources there's no xxx-large, although using it
            # renders fine but fails to be saved in the HTML
            if (hLevel != level):
                # Don't bother using setHeadingLevel since it's not supported on
                # this Qt version and as per docs it doesn't change the font
                # size anyway, it's just a signaling flag
                fmt.setProperty(QTextFormat.FontSizeAdjustment, 3 - level + 1)
                fmt.setFontWeight(QFont.Bold)

                # Set the charformat as both the charformat and the
                # blockcharformat, this makes the format work for non-empty
                # blocks (which selected text take the format from the
                # charformat) and empty blocks (which new text will take the
                # format from the blockcharformat)
                cursor.setBlockCharFormat(fmt)
                cursor.setCharFormat(fmt)

                self.setTextCursor(cursor)

                # Insert a horizontal ruler if there's no next block or if it's
                # not a ruler already, needs to be on an empty block by itself
                # (setting a ruler on a block with text renders fine but when
                # saving to HTML saves the HR but not the text)
                nextBlock = cursor.block().next()
                if ((not nextBlock.isValid()) or 
                    (nextBlock.blockFormat().property(QTextFormat.BlockTrailingHorizontalRulerWidth) is None)):
                    blockFmt = cursor.blockFormat()
                    blockFmt.setProperty(QTextFormat.BlockTrailingHorizontalRulerWidth, True)

                    # No need to set size and bold on the ruler, remove so they
                    # don't get propagated to new blocks if this is the last
                    # block
                    fmt = cursor.blockCharFormat()
                    fmt.setFontWeight(QFont.Normal)
                    fmt.clearProperty(QTextFormat.FontSizeAdjustment)
                    
                    cursor.clearSelection()
                    cursor.insertBlock(blockFmt, fmt)

                    # If this is the last block in the document, insert an empty
                    # block so the ruler is not propagated to the next block

                    # XXX Alternatively clear all formatting when return is
                    #     pressed on empty line?
                    if (not cursor.block().next().isValid()):
                        blockFmt = cursor.blockFormat()
                        blockFmt.clearProperty(QTextFormat.BlockTrailingHorizontalRulerWidth)

                        cursor.insertBlock(blockFmt)
                    
                # Put the cursor back to where it was
                # XXX This doesn't work when the cursor is at the end of the
                #     line (cursor ends in the ruler block) but it works when
                #     the cursor is elsewhere?
                self.setTextCursor(prevCursor)
                
            else:
                # Already this heading level so toggle, clear formatting and
                # ruler

                # setProperty(None) instead of clearProperty also works
                fmt.clearProperty(QTextFormat.FontSizeAdjustment)
                fmt.setFontWeight(QFont.Normal)
                
                # Note mergeCharFormat fails to remove FontSizeAdjustment, use
                # setCharFormat instead
                cursor.setCharFormat(fmt)
                cursor.setBlockCharFormat(fmt)
                self.setTextCursor(cursor)

                # Remove the selection
                self.setTextCursor(prevCursor)
                
                nextBlock = cursor.block().next()
                # Note the ruler is set on the blockFormat, not on the first
                # textFormat, in fact the block has no textFormats since it's 
                # empty
                if ((nextBlock is not None) and 
                    (nextBlock.blockFormat().property(QTextFormat.BlockTrailingHorizontalRulerWidth) is not None)):
                    cursor = QTextCursor(nextBlock)
                    cursor.select(QTextCursor.BlockUnderCursor)
                    cursor.removeSelectedText()
                    cursor.deleteChar()

        elif ((formatChar in ["t", "q"]) or formatChar.startswith("l")):
            # Already done
            pass

        else:
            assert False, "Unrecognized format char %r!!!" % formatChar

        if (formatChar in ["b", "i", "u", "q"]):
            # setCurrentCharFormat will change the selectionformat or, if no
            # selection, the format for the next typed chars
            self.setCurrentCharFormat(fmt)
                
    def parseFormat(self, position, length):
        """
        Parse markdown-like formatting tokens:

        #, ##, ###, ####        Headings
        - + * 1. I. i. a. A.    Different bullet lists (disc, square, circle, etc)
        >                       Blockquote

        xXX Missing the following:
        
        **<nonspace> as begin bold
        <nonspace>** as end bold
        *<nonspace> as begin italic
        <nonspace>* as end italic
        ***<nonspace> as begin italic bold
        <nonspace>*** as end italic bold
        _<nonspace> as begin underline
        <nonspace>_ as end underline
        
        -- horizontal rule
        [title](link)

        del on empty line, remove horizontal rule if any
        backspace on blockquote remove blockquote
        
        nest lists on tab inside list
        
        See https://www.markdownguide.org/cheat-sheet/
        """
        # The token must start at begin of block/paragraph, find the start of
        # the block where the chars are being added. Note QTextEdit uses
        # u"\u2029" or QChar.ParagraphSeparator instead of \n
        doc = self.document()
        block = doc.findBlock(position)

        # Parse all tokens added
        # XXX This shouldn't work for multiple tokens because toggleFormat will
        #     add blocks, needs toggleFormat to render to an "offscreen
        #     textfragment" that is later pasted wholesome into the document, or
        #     needs to parse tokens starting with the last one. Not clear why
        #     it's working in some limited testing.
        while (block.isValid() and (block.position() < position + length)):
            tokenStart = tokenEnd = block.position()
            token = ""
            # XXX This could ignore parsing the token if the position is far
            #     away from the block start, since all tokens must start at
            #     beginning of paragraph and are at most a few chars long
            while ((tokenEnd < (tokenStart + block.length())) and (not doc.characterAt(tokenEnd).isspace())):
                c = doc.characterAt(tokenEnd)
                token += c
                tokenEnd += 1
            
            # Note tokenEnd is exclusive, python string style
            assert len(token) == (tokenEnd - tokenStart)

            block = block.next()
        
            logger.info("token is %r prechar is %r", token, doc.characterAt(tokenStart - 1))
            formatChar = None
            if (token.startswith("#")):
                count = min(token.count("#"), 4)
                formatChar = "%d" % count
                
            elif (token == ">"):
                # XXX Missing handling tab/shift-tab to increase/decrease
                #     indent?
                formatChar = "q"

            elif (token in VTTTextEditor.tokenToListStyle):
                # XXX When parsing multiple tokens this should merge all the 
                #     list items in the same list
                # XXX Missing nesting indentation levels with tab, removing
                #     with shift+tab
                styles = sorted(VTTTextEditor.tokenToListStyle.values())
                style = VTTTextEditor.tokenToListStyle[token]
                i = index_of(styles, style)
                formatChar = "l" * (i + 2)
            
            if (formatChar is not None):
                # Remove token plus space
                cursor = self.textCursor()
                cursor.setPosition(tokenStart)
                cursor.setPosition(tokenEnd + 1, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                # Move the cursor to the format toggling point
                self.setTextCursor(cursor)
                self.toggleFormat(formatChar)

            # XXX Have a console mode to enter commands, die rolls, macros, etc?
            # XXX Allow html?
        

    def contentsChange(self, position, charsRemoved, charsAdded):
        # For live keyboard entry this is called with a single char added or
        # removed, for cut and paste this is called with several
        logger.info("position %d removed %d added %d", position, charsRemoved, charsAdded)

        # XXX This only considers tokens as result of adding chars, what if
        #     deleting chars causes a token to be formed?
        if (charsAdded > 0):
            cursor = self.textCursor()
            cursor.setPosition(position)
            cursor.setPosition(position + charsAdded, QTextCursor.KeepAnchor)
            chars = cursor.selectedText()

            logger.info("added chars are %r", chars)

            # Tokens must start at beginning of paragraph and end with space, 
            # early exit if no spaces were added
            if (" " not in chars):
                return

            self.parseFormat(position, charsAdded)


    def canInsertFromMimeData(self, source):
        if (source.hasImage()):
            return True

        else:
            return super(VTTTextEditor, self).canInsertFromMimeData(source)

    def insertFromMimeData(self, source):
        # See https://www.pythonguis.com/examples/python-rich-text-editor/

        cursor = self.textCursor()
        document = self.document()

        # XXX Allow internal urls like qtvtt:combattracker, qtvtt:image to
        #     display live tables, maps, images, etc
        if (source.hasUrls()):
            # XXX This path is untested and left for reference on how to insert
            #     images via resources (which need to be saved separately)
            #     instead of embedded as dataurl
            # XXX Get these from Qt
            IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".webp"]
            for url in source.urls():
                # XXX use toLocalFile in several other places that convert from
                #     QUrl to file?
                
                # XXX Images get pasted as html img tags with the provided url,
                #     the image needs to be saved separately or converted to
                #     an embedded data: url
                ext = os_path_ext(str(u.toLocalFile()))
                ext = ext.lower()
                if (url.isLocalFile() and (ext in IMAGE_EXTENSIONS)):
                    image = QImage(url.toLocalFile())
                    document.addResource(QTextDocument.ImageResource, url, image)
                    cursor.insertImage(url.toLocalFile())

                else:
                    super(VTTTextEditor, self).insertFromMimeData(source)

        elif (source.hasImage()):
            image = source.imageData()
            imformat = "PNG"
            # XXX This could use a local url instead and save the file
            #     separately?
            dataUrl = qImageToDataUrl(image, imformat)
            # XXX Needs scaling settings, note Qt only allows absolute sizes,
            #     not relative (percentage) sizes which would need to hook on
            #     QTextEdit resizing support and wouldn't be saved with the 
            #     html?
            cursor.insertImage(dataUrl)
            
        else:
            super(VTTTextEditor, self).insertFromMimeData(source)

            
class DocEditor(QWidget):
    """
    Documentation text editor, search function, and realtime-generated table of
    contents.
    """
    def __init__(self, *args, **kwargs):
        super(DocEditor, self).__init__(*args, **kwargs)

        self.lastCursor = None
        self.lastPattern = None
        self.lastExtraSelection = 0

        tocTree = QTreeWidget()
        self.tocTree = tocTree
        tocTree.setColumnCount(1)
        tocTree.setHeaderHidden(True)
        # Don't focus this on wheel scroll
        tocTree.setFocusPolicy(Qt.StrongFocus)

        def treeItemClicked(item):
            logger.info("%s", item.text(0))
            position = item.data(0, Qt.UserRole)
            textCursor = self.textEdit.textCursor()
            textCursor.setPosition(item.data(0, Qt.UserRole))
            # Go to the end then back so the line is not at the bottom of the
            # viewport (both ensureCursorVisible and setCursor display the line
            # at the bottom of the viewport)
            self.textEdit.moveCursor(QTextCursor.End)
            self.textEdit.setTextCursor(textCursor)
            self.textEdit.setFocus(Qt.TabFocusReason)
            
        tocTree.itemClicked.connect(treeItemClicked)

        textEdit = VTTTextEditor()
        self.textEdit = textEdit
        # Don't focus this on wheel scroll
        textEdit.setFocusPolicy(Qt.StrongFocus)

        # XXX Add toolbar buttons for format, font name, font size, alignment,
        #     use the current statusbar code to update them?

        def textChanged():
            # Rebuild the TOC
            # XXX This should only rebuild if chars in headers have changed?
            tocTree.clear()
            
            stack = []
            parentItem = None
            item = None
            prevLevel = 0
            # The non-H1 font sizes appear in the html as
            #
            #   re.finditer(r"<span [^>]*font-size:(\d+)pt[^>]*>([^<]*)", html):
            #
            # but unless anchors are used in the text, it's hard to find a
            # reference to scroll to when the TOC item is clicked, so parse
            # blocks instead of html.
            #
            # XXX This could also use insertHtml and anchors with name attribute and
            #     then scrollToAnchor
            #     https://stackoverflow.com/questions/20678610/qtextedit-set-anchor-and-scroll-to-it
            
            # Note that eg each cell in a table is a different block, so there
            # can be lots of blocks in complex documents
            # XXX Sync the TOC from the text too
            block = textEdit.document().begin()
            while (block.isValid()):
                # There can be multiple textformats per block, but for headings
                # only need to look at the first one (note block.charFormatIndex
                # or block.charFormat give the default format for the block, not 
                # for the chars inside the block)
                textFormats = block.textFormats()
                if (len(textFormats) > 0):
                    text = block.text()
                    position = block.position()
                    fontSizeAdjustment = textFormats[0].format.property(QTextFormat.FontSizeAdjustment)
                    if (fontSizeAdjustment is None):
                        level = None

                    else:
                        level = 3 + 1 - fontSizeAdjustment
                
                else:
                    # Some blocks have no textFormats, skip this block
                    level = None
                block = block.next()
                
                if (level is None):
                    continue
                
                deltaLevels = prevLevel - level

                # Find a parent for this item
                if (deltaLevels > 0):
                    # This item is higher in the tree hierarchy than the current
                    # level, pop as many parents as deltaLevels
                    parentItem = stack[-deltaLevels]
                    stack = stack[:-deltaLevels]
                    
                elif (deltaLevels < 0):
                    # This item is lower in the tree hierarchy than the current
                    # level, push as many parents as deltaLevels
                    stack.append(parentItem)
                    parentItem = item
                    # If there are missing parents, create as many dummy
                    # placeholder parents as deltaLevels - 1
                    for i in xrange(-deltaLevels - 1):
                        stack.append(parentItem)
                        item = QTreeWidgetItem(["???"])
                        if (parentItem is None):
                            tocTree.addTopLevelItem(item)

                        else:
                            parentItem.addChild(item)
                            parentItem.setExpanded(True)
                        parentItem = item

                # Create the item with the found parent
                item = QTreeWidgetItem([text])
                if (parentItem is None):
                    tocTree.addTopLevelItem(item)

                else:
                    parentItem.addChild(item)
                    parentItem.setExpanded(True)

                # Store a reference so it can scroll there when activating the
                # tree item
                item.setData(0, Qt.UserRole, position)
                
                prevLevel = level
                
        textEdit.textChanged.connect(textChanged)
        textEdit.installEventFilter(self)

        lineEdit = QLineEdit(self)
        self.lineEdit = lineEdit
        lineEdit.setPlaceholderText("Search...")
        lineEdit.textChanged.connect(self.textChanged)
        lineEdit.returnPressed.connect(self.returnPressed)
        lineEdit.installEventFilter(self)

        label = QLabel("0/0")
        self.counterLabel = label

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(QLabel("Search"))
        hbox.addWidget(lineEdit)
        # XXX Make replace, word, case and regexp work, will need better
        #     handling of the current hide on focusout policy
        hbox.addWidget(QLabel("Replace"))
        lineEdit = QLineEdit()
        lineEdit.setEnabled(False)
        hbox.addWidget(lineEdit)
        for name in ["word", "case", "regexp"]:
            checkbox = QCheckBox(name)
            checkbox.setEnabled(False)
            hbox.addWidget(checkbox)
        hbox.addWidget(label)
        searchBox = QWidget()
        self.searchBox = searchBox
        searchBox.setLayout(hbox)
        searchBox.setVisible(False)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(textEdit)
        vbox.addWidget(searchBox)
        
        hsplitter = QSplitter(Qt.Horizontal)
        self.hsplitter = hsplitter
        hsplitter.addWidget(tocTree)
        widget = QWidget()
        widget.setLayout(vbox)
        hsplitter.addWidget(widget)
        hsplitter.setStretchFactor(1, 20)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(hsplitter)

        self.setLayout(vbox)

        # Always focus on the textEdit first when the DocEditor receives focus
        self.setTabOrder(textEdit, hsplitter)
        
    def eventFilter(self, source, event):
        logger.info("source %s, event %s", class_name(source), EventTypeString(event.type()))
        if ((source in [self.textEdit, self.lineEdit]) and qEventIsShortcutOverride(event, "ctrl+f")):
            logger.info("ShortcutOverride for text %r key %d", event.text(), event.key())
            event.accept()
            return True

        elif ((source == self.textEdit) and qEventIsKeyShortcut(event, "ctrl+f")):
            # XXX Handle F3 as continue search
            self.searchBox.setVisible(True)
            self.lineEdit.setFocus(Qt.TabFocusReason)
            # If there's no selection, select the current word
            textCursor = self.textEdit.textCursor()
            if (not textCursor.hasSelection()):
                textCursor.select(QTextCursor.WordUnderCursor)
                # Note no need to setTextCursor, not clear why
            
            if (textCursor.hasSelection()):
                self.lineEdit.setText(textCursor.selectedText())

            else:
                # Do a double change to trigger textChanged and start the search
                # (single setText to the same vale won't trigger textChanged)
                text = self.lineEdit.text()
                self.lineEdit.setText("")
                self.lineEdit.setText(text)
            self.lineEdit.selectAll()

            return True

        elif ((source == self.lineEdit) and qEventIsKeyShortcut(event, "esc")):
            self.textEdit.setFocus(Qt.TabFocusReason)
            return True

        elif ((source == self.lineEdit) and (event.type() == QEvent.FocusOut)):
            self.searchBox.setVisible(False)
            # XXX This should select the selection on the textedit?
            self.textEdit.setExtraSelections([])

        return False

    def findNext(self, next = True):
        logger.info("")
        selections = self.textEdit.extraSelections()
        delta = 1 if next else -1
        if (0 <= (self.lastExtraSelection + delta) < len(selections)):
            # Yellow the previous position, orange the next
            selection = selections[self.lastExtraSelection]
            selection.format.setBackground(Qt.yellow)
            self.lastExtraSelection += delta
            selection = selections[self.lastExtraSelection]
            selection.format.setBackground(QColor("orange"))

            self.textEdit.setExtraSelections(selections)
            self.counterLabel.setText("%d/%d" % (self.lastExtraSelection + 1, len(selections)))

            # Move to the selection
            textCursor = QTextCursor(selection.cursor)
            textCursor.clearSelection()
            self.textEdit.setTextCursor(textCursor)

    def returnPressed(self):
        # XXX This is almost replicated with DocBrowser, should have a readonly
        #     TextEditor and refactor?
        logger.info("modifiers 0x%x ctrl 0x%x shift 0x%x ", 
            int(qApp.keyboardModifiers()),
            int(qApp.keyboardModifiers() & Qt.ControlModifier),
            int(qApp.keyboardModifiers() & Qt.ShiftModifier)
        )
        next = (int(qApp.keyboardModifiers() & Qt.ShiftModifier) == 0)
        self.findNext(next)

    def textChanged(self, s):
        logger.info("%s", s)
        # XXX This is replicated in DocBrowser, create a readonly searcheable
        #     textedit and refactor
        pattern = str.join("|", self.lineEdit.text().split())
        selections = []
        color = QColor("orange")
        # XXX This should start searching from the current cursor position, not
        #     from the start of the document
        self.textEdit.moveCursor(QTextCursor.Start)
        while (True):
            found = self.textEdit.find(QRegExp(pattern, Qt.CaseInsensitive))
            textCursor = self.textEdit.textCursor()
            if (not found):
                # Clear the selection, go to the first occurrence, end
                if (len(selections) > 0):
                    textCursor = QTextCursor(selections[0].cursor)
                    textCursor.clearSelection()
                    self.textEdit.setTextCursor(textCursor)
                    self.lastExtraSelection = 0
                break

            # Use extra selections to highlight matches, this has several
            # advantages vs. using real formatting (doesn't trigger the
            # modification flag, no need to block signals, not saved with the
            # document, simpler code although probably less efficient since all
            # the extra selections must be set everytime when traversing matches
            # vs. only the previous and current match if using regular
            # formatting)
            selection = QTextEdit.ExtraSelection()
            selection.cursor = textCursor
            fmt = textCursor.charFormat()
            fmt.setBackground(color)
            selection.format = fmt
            selections.append(selection)
            color = Qt.yellow

        self.textEdit.setExtraSelections(selections)
        self.counterLabel.setText("%d/%d" % (self.lastExtraSelection + 1, len(selections)))

    def modified(self):
        return self.textEdit.document().isModified()

    def setModified(self, modified):
        self.textEdit.document().setModified(modified)
        
    def getFilepath(self):
        # XXX Is there any advantage in using metaInformation vs. some new field?
        url = self.textEdit.document().metaInformation(QTextDocument.DocumentUrl)
        if (url == ""):
            return None
        url = QUrl(url)
        return url.toLocalFile()

    def setFilepath(self, filepath):
        if (filepath is not None):
            url = QUrl.fromLocalFile(filepath)
            url = url.toString()

        else:
            url = None
        self.textEdit.document().setMetaInformation(QTextDocument.DocumentUrl, url)

    def clearText(self):
        logger.info("")
        self.textEdit.clear()
        self.setFilepath(None)

    def loadText(self, filepath):
        logger.info("%s", filepath)
        
        with open(filepath, "rb") as f:
            content = f.read()
            self.textEdit.setHtml(content)
            self.setFilepath(filepath)

    def saveText(self, filepath = None):
        if (filepath is not None):
            self.setFilepath(filepath)
        filepath = self.getFilepath()
        if (filepath is None):
            logger.warning("Ignoring saving on empty filepath")
            return

        logger.info("saving %r", filepath)
        # XXX This can be saved to other formats using QTextDocumentWriter
        #     https://www.qtcentre.org/threads/28550-QTextDocument-not-keeping-its-format-when-saving-to-ODF-file
        #     But there's no QTextDocumentReader
        #     See https://stackoverflow.com/questions/31958553/qtextdocument-serialization
        contents = self.textEdit.document().toHtml("utf-8")
        with open(filepath, "wb") as f:
            f.write(contents.encode("utf-8"))


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
        self.lineEdit = lineEdit
        lineEdit.setPlaceholderText("Search...")
        listWidget = QListWidget()
        self.listWidget = listWidget
        
        tocTree = QTreeWidget()
        self.tocTree = tocTree
        tocTree.setColumnCount(1)
        tocTree.setHeaderHidden(True)
        
        self.lastCursor = None
        self.lastPattern = None
        self.curTocFilePath = None
        self.curTocItem = None
        self.sourceTitle = ""
        
        # XXX Have browser zoom, next, prev buttons / keys, font?
    
        textEdit = QTextBrowser()
        self.textEdit = textEdit

        # Don't focus these on wheel scroll
        textEdit.setFocusPolicy(Qt.StrongFocus)
        listWidget.setFocusPolicy(Qt.StrongFocus)
        tocTree.setFocusPolicy(Qt.StrongFocus)

        lineEdit.textChanged.connect(self.textChanged)
        lineEdit.returnPressed.connect(self.returnPressed)
        # XXX This should open in a new browser/tab if ctrl is pressed the same
        #     way the monster browser does
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
        logger.info("modifiers 0x%x ctrl 0x%x shift 0x%x ", 
            int(qApp.keyboardModifiers()),
            int(qApp.keyboardModifiers() & Qt.ControlModifier),
            int(qApp.keyboardModifiers() & Qt.ShiftModifier)
        )
        # PyQt complains if 0 is passed as findFlags to find, use explicit
        # conversion
        findFlags = QTextDocument.FindBackward if (int(qApp.keyboardModifiers() & Qt.ShiftModifier) != 0) else QTextDocument.FindFlag(0)

        # Yellow the previously current position
        # XXX Use extraSelections instead of straight formatting
        fmt = self.lastCursor.charFormat()
        fmt.setBackground(Qt.yellow)
        self.lastCursor.setCharFormat(fmt)
        self.textEdit.setTextCursor(self.lastCursor)

        if ((int(qApp.keyboardModifiers() & Qt.ControlModifier) != 0) or 
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
        # XXX Use extraSelections instead of straight formatting
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

# This is a debug setting to disable the player view in order to get higher 
# perf when editing walls, etc
g_disable_player_view = False
# Image bytes (eg PNG or JPEG) between the app thread and the http server thread
# XXX This needs to be more flexible once there are multiple images shared, etc
g_img_bytes = None
# Encoding to PNG takes 4x than JPEG, use JPEG (part of http and app handshake
# configuration)
# XXX For consistency the html and the path should use image.jpg instead of .png
#     when encoding to jpeg (using the wrong extension is ok, though, because
#     the extension is ignored when a mime type is passed)
#imctype = "image/png"
imctype = "image/jpeg"
imformat = "PNG" if imctype.endswith("png") else "JPEG"
# XXX This is shared between the app thread and the http server thread, should
#     have some kind of locking (although most of the time it's not needed
#     because of the GIL)
g_handouts = []
g_server_ips = []
class VTTHTTPRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        if (self.path.startswith("/image.png")):
            logger.debug("get start %s", self.path)

            if (g_img_bytes is not None):
                logger.debug("wrapping in bytes")
                f = io.BytesIO(g_img_bytes)
                clength = len(g_img_bytes)

            else:
                # This can happen at application startup when g_img_bytes is not
                # ready yet
                # XXX Fix in some other way?
                logger.debug("null g_img_bytes, sending empty")
                f = io.BytesIO("")
                clength = 0

            logger.debug("returning")
            ctype = imctype

        elif (self.path == "/fog.svg"):
            ctype = "image/svg+xml"
            clength = os.path.getsize("_out/fog.svg")

            f = open("_out/fog.svg", "rb")

        elif (self.path == "/"):
            # XXX Share character sheets when available
            # XXX This could have some message or status or "the story until now"

            # XXX The path probably needs some timestamp so it doesn't get cached?

            # XXX Get this from the campaign/scene
            title = "Title goes here"
            
            body_lines = []

            body_lines.append("<H1>%s</H1>" % title)

            # Maps
            body_lines.append("Scenes")
            body_lines.append("<ul>")
            # XXX Hardcoded for the time being
            map_name = "map"
            map_url = "/index.html"
            body_lines.append('<li><a href="%s">%s</a>' % (map_url, map_name))
            body_lines.append("</ul>")

            # Handouts
            body_lines.append("Handouts")
            # XXX There should also be campaign-level handouts that appear in
            #     all scenes
            # XXX This should have folders for manuals, etc
            # XXX Add thumbnails/file type icon
            body_lines.append("<ul>")
            for handout in g_handouts:
                if (handout.shared):
                    body_lines.append('<li><a href="%s">%s</a>' % (handout.filepath, handout.name))
            body_lines.append("</ul>")
            
            html = "<html><title>%s</title><body>%s</body></html>" % (title, str.join("\n", body_lines))
            html = html.encode('ascii', 'xmlcharrefreplace')
            
            f = StringIO.StringIO(html)
            ctype = "text/html"
            clength = len(html)

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
            # Only allow the files in handouts otherwise
            # SimpleHTTPRequestHandler would expose all the files in the
            # directory
            filepath = urllib.unquote(self.path[1:])
            filepath = filepath.replace("/", os.sep)
            if (filepath in [handout.filepath for handout in g_handouts if handout.shared]):
                # Do a explicit call, can't use super since BaseRequestHandler
                # doesn't derive from object
                return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

            else:
                self.send_error(404)
                return 

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
    global g_server_ips
    g_server_ips = socket.gethostbyname_ex(socket.gethostname())
    
    # XXX In case some computation is needed between requests, this can also do
    #     httpd.handle_request in a loop
    httpd.serve_forever()

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
        
        self.viewport().installEventFilter(self)

        # Set a dummy pixmap in case player view is disabled
        pix = QPixmap(QSize(1, 1))
        self.setPixmap(pix)

        self.setCursor(Qt.OpenHandCursor)

    def setBackgroundColor(self, color):
        self.viewport().setStyleSheet("background-color:rgb%s;" % (qtuple(color),))

    def setImage(image):
        if (image.colorSpace().isValid()):
            image.convertToColorSpace(QColorSpace.SRgb)
        self.setPixmap(QPixmap.fromImage(image))

    def setPixmap(self, pixmap):
        logger.info("size %s", pixmap.size())
        self.imageLabel.setPixmap(pixmap)
        # Note this doesn't update the widget, the caller should call some
        # resizing function to update
        
    def zoomImage(self, zoomFactor, anchor = QPointF(0.5, 0.5)):
        logger.debug("factor %s anchor %s", zoomFactor, anchor)
        self.scale *= zoomFactor
        self.resizeImage(self.scale * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.horizontalScrollBar(), zoomFactor, anchor.x())
        self.adjustScrollBar(self.verticalScrollBar(), zoomFactor, anchor.y())

    def adjustScrollBar(self, scrollBar, factor, anchor):
        logger.debug("anchor %s value %s page %s min,max %s", anchor, scrollBar.value(), scrollBar.pageStep(), (scrollBar.minimum(), scrollBar.maximum()))

        # anchor bottom
        #scrollBar.setValue(int(scrollBar.value() * factor - scrollBar.pageStep() + factor * scrollBar.pageStep()))
        # anchor top
        #scrollBar.setValue(int(scrollBar.value() * factor))
        # aanchor linear interpolation wrt anchor:
        scrollBar.setValue(int(scrollBar.value() * factor + anchor * (factor * scrollBar.pageStep() - scrollBar.pageStep())))
        
    def setFitToWindow(self, fitToWindow):
        logger.info("%s", fitToWindow)
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
        logger.info("size %s fitToWindow %s", size, self.fitToWindow)
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
        logger.info("size %s", event.size())

        if (self.fitToWindow):
            newSize = event.size()
            self.resizeImage(newSize)
        
        super(ImageWidget, self).resizeEvent(event)

    def mousePressEvent(self, event):
        logger.info("pos %s", event.pos())

        if (event.button() == Qt.LeftButton):
            self.setCursor(Qt.ClosedHandCursor)
            self.prevDrag = event.pos()

        else:
            super(ImageWidget, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        logger.info("pos %s", event.pos())

        if (event.button() == Qt.LeftButton):
            self.setCursor(Qt.OpenHandCursor)
        
        else:
            super(ImageWidget, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        logger.debug("pos %s", event.pos())

        if (event.buttons() == Qt.LeftButton):
            delta = (event.pos() - self.prevDrag)
            logger.debug("Scrolling by %s", delta)
            
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
                
            self.prevDrag = event.pos()

        else:

            return super(ImageWidget, self).mouseMoveEvent(event)

    def eventFilter(self, source, event):
        logger.debug("source %s type %s", class_name(source), EventTypeString(event.type()))
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

def load_test_tokens(cell_diameter):
    # Create some dummy tokens for the time being
    # XXX Remove once tokens can be imported or dragged from the token browser
    map_tokens = []
    for filename in [
        "Hobgoblin.png", "Hobgoblin.png", "Goblin.png" ,"Goblin.png", 
        "Goblin.png", "Gnoll.png", "Ogre.png", "Ancient Red Dragon.png", 
        "Knight.png", "Priest.png", "Mage.png"][0:0]:
        map_token = Struct()
        map_token.filepath = os.path.join("_out", "tokens", filename)
        map_token.scene_pos = (0.0, 0.0)
        map_token.name = os_path_name(filename)
        map_token.hidden = False
        map_token.ruleset_info = Struct(**default_ruleset_info)
        
        pix_scale = cell_diameter
        # XXX This should check the monster size or such
        if ("Dragon" in filename):
            pix_scale *= 4.0
        elif ("Ogre" in filename):
            pix_scale *= 1.5

        map_token.scale = pix_scale

        map_tokens.append(map_token)

    return map_tokens

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

    scene = Struct()
    import_ds(scene, ds_filepath, map_filepath, img_offset_in_cells)

    return scene


class GraphicsRectNotifyItem(QGraphicsRectItem):
    def itemChange(self, change, value):
        assert None is logger.debug("change %s value %s", GraphicsItemChangeString(change), value)
        # Scene can be none before the item is added to the scene
        if ((self.scene() is not None) and
             # XXX This is in the hotpath of updateFog/setWallsVisible, and it's
             #     not clear why VTTGraphicsScene.itemChanged calls updateFog
             #     even if updates are locked in setWallsVisible, but appears in
             #     the profiling logs
             (change not in [QGraphicsItem.ItemZValueHasChanged, QGraphicsItem.ItemZValueChange])):
             return self.scene().itemChanged(self, change, value)

        else:
            return super(GraphicsRectNotifyItem, self).itemChange(change, value)


class GraphicsPixmapNotifyItem(QGraphicsPixmapItem):
    """
    Create a notify item since monkey patching itemChange 
        QGraphicsPixmapItem.itemChange = lambda s, c, v: logger.info("itemChange %s", v)
    doesn't work
    """

    def sceneEventFilter(self, watched, event):
        # The main token item receives events from the label item in order to
        # detect when label editing ends
        logger.debug("watched %s type %s", class_name(watched), EventTypeString(event.type()))

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
                result = True

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

        Return parent + children boundingRect, this may give issues if any code
        assumes the center of the bounding rect matches the center of the
        pixmap, since this boundingRect includes the label bounding rect.

        Also, the focus/selected rect code will use these bounds to draw the
        dotted rectangle, so it may not be desirable to return the parent +
        children?
        """
        
        rect = super(GraphicsPixmapNotifyItem, self).boundingRect()
        rect |= self.childrenBoundingRect()
        logger.info("%s %s", self.data(0).name, qtuple(rect))
        return rect

    def itemChange(self, change, value):
        logger.debug("change %s value %s", GraphicsItemChangeString(change), value)
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
        self.wallHandleItems = set()
        self.imageItems = set()
        self.gridLineItems = set()
        self.map_scene = map_scene

        self.blendFog = False
        self.fogVisible = False
        self.lockDirtyCount = 0
        self.dirtyCount = 0
        self.fogPolysDirtyCount = -1

        self.allWallsItem = QGraphicsItemGroup()
        self.allDoorsItem = QGraphicsItemGroup()
        self.gridItem = QGraphicsItemGroup()
        self.gridHandleItem = None
        self.fogItem = QGraphicsItemGroup()
        self.fogCenter = None
        self.fogCenterLocked = False
        self.fog_polys = []
        self.fog_mask = None
        self.fogColor = None

        self.cellDiameter = 70
        self.cellAnchor = None
        self.cellOffset = QPointF(0, 0)
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
        if (not self.dirtyCountIsLocked()):
            self.dirtyCount += 1

    def dirtyCountIsLocked(self):
        return (self.lockDirtyCount > 0)

    def setLockDirtyCount(self, locked):
        self.lockDirtyCount += 1 if locked else -1
        assert self.lockDirtyCount >= 0

    def setWallHandleItemPos(self, wallHandleItem, pos):
        """
        Update the wall points wrt to the wall handle position
        """
        scene = self.map_scene

        wallHandleItem.setPos(pos)
        p = wallHandleItem.pos()
        wallItem = self.getWallItemFromWallHandleItem(wallHandleItem)
        i = self.getPointIndexFromWallHandleItem(wallHandleItem)
        map_wall = wallItem.data(0)
        map_wall.points[i] = [p.x(), p.y()]

        qpp = QPainterPath()
        qpoints = [QPointF(*p) for p in map_wall.points]
        if (map_wall.closed):
            qpoints.append(qpoints[0])
        qpp.addPolygon(QPolygonF(qpoints))
        wallItem.setPath(qpp)

        # Clear fog of war since changing the walls makes fog of war
        # (acumulated fogs) look funny
        self.fog_mask = None

    def snapPosition(self, pos, snapToGrid, fineSnapping):
        logger.debug("pos %s snapToGrid %s fineSnapping %s", qtuple(pos), snapToGrid, fineSnapping)

        snapPos = pos
        if (snapToGrid):
            snapGranularity = self.cellDiameter / 2.0
            if (fineSnapping):
                snapGranularity /= 10.0
            
            snapOffset = self.cellOffset
            snapPos = pos - snapOffset
            snapPos = (snapPos / snapGranularity).toPoint() * snapGranularity 
            snapPos += snapOffset
            logger.info("Snapping %s to %s", qtuple(pos), qtuple(snapPos))
            
        return snapPos

    def snapPositionWithCurrentSettings(self, pos):
        return self.snapPosition(pos, self.snapToGrid, 
            int(qApp.keyboardModifiers() & Qt.ShiftModifier) != 0)

    def itemChanged(self, item, change, value):
        """
        This receives notifications from GraphicsPixmapNotifyItem/GraphicsRectNotifyItem
        """
        logger.debug("change %s", GraphicsItemChangeString(change))

        if (change == QGraphicsItem.ItemPositionChange):
            
            if (self.isGridHandle(item)):
                snapPos = value
                
                # If Ctrl is pressed, check that:
                # - The value is snapped so both x,y have the same value
                # - The value is not smaller than a given minimum grid cell size
                # XXX Ideally this should set a cell size around 70, values of
                #     2000 cause high scale values (eg 7000) that significantly
                #     slow down unnecessarily, but probably due to scenerect
                #     cascading into the imagewidget and fog of war mask sizes
                #     so fix there?
                if (int(qApp.keyboardModifiers() & Qt.ControlModifier) != 0):
                    # XXX This is not grid-snapping for mouse resizing, should
                    #     it? (grid snapping for keyboard resizing is done by
                    #     virtue of increasing by snapping deltas at keypress
                    #     time)
                    # cellAnchor is the topleft corner of the grid cell where
                    # the gridHandle is
                    delta = value - self.cellAnchor 
                    # Use the delta in y and ignore the delta in x, this is so
                    # keyboard resizing works (which can only delta in y since
                    # ctrl+left/right is swallowed by the media player) as well
                    # as mouse resizing (which deltas in both x and y)
                    cellDiameter = delta.y()
                    # Don't let the cellDiameter become smaller than a minimum
                    if (cellDiameter <= 5):
                        logger.info("Rejecting small cellDiameter")
                        return item.pos()
                    snapPos = self.cellAnchor + QPointF(cellDiameter, cellDiameter)
                    
            else:
                # value is the new position, snap 
                snapPos = self.snapPositionWithCurrentSettings(value)

            return snapPos

        if (change == QGraphicsItem.ItemPositionHasChanged):
            if (self.isToken(item)):
                item.data(0).scene_pos = qlist(item.scenePos())

            elif (self.isWallHandle(item)):
                # Position has already been updated, just pass current position
                # to update the wall points
                self.setWallHandleItemPos(item, item.pos())

            elif (self.isGridHandle(item)):
                # Remove the grid
                self.removeGrid()
                # Set the cell diameter
                # XXX What items should changing the cell diameter affect?
                #     Probably not positions but token sizes? Have a global
                #     scale on the scene dependent on the grid diameter? (but
                #     that would change positions too). 
                # XXX Checking modifiers here is not a good idea?
                if (int(qApp.keyboardModifiers() & Qt.ControlModifier) != 0):
                    # This has already been snapped to make x and y sizes to
                    # match, just use one of them to calculate the new diameter
                    cellOffset = self.cellOffset
                    cellDiameter = item.pos().x() - self.cellAnchor.x()
                    cellDiameter = abs(cellDiameter)
                    
                else:
                    cellDiameter = self.cellDiameter
                    cellOffset = QPointF(item.pos().x() % cellDiameter, item.pos().y() % cellDiameter)

                self.setCellOffsetAndDiameter(cellOffset, cellDiameter)
                # XXX This is not clean, but needs to restore cellAnchor
                #     otherwise it gets reset to the "canonical" position
                self.cellAnchor = item.pos() - QPointF(cellDiameter, cellDiameter)
                
                # Dirty the scene (cell diameter or offset changed, so any item
                # that depends on that should change) and the grid
                self.makeDirty()
                self.addGrid()
                
                
        if (change in [
            QGraphicsItem.ItemVisibleHasChanged, 
            QGraphicsItem.ItemPositionHasChanged,
            # Need to update fog if the focus changes
            QGraphicsItem.ItemSelectedHasChanged,
            ]):
            if (not self.dirtyCountIsLocked()):
                self.makeDirty()
                if (self.getFogCenter() is not None):
                    self.updateFog()
        
        return value

    def getFogCenter(self):
        logger.info("")
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
        
    def setCellOffsetAndDiameter(self, cellOffset, cellDiameter):
        self.cellOffset = cellOffset
        self.cellDiameter = cellDiameter
        # XXX This assumes the grid image is set at 0,0, should get the image
        #     rect to offset it (but it needs to exist at this time, which it may
        #     not at initialization?)
        self.cellAnchor = self.cellOffset
        self.map_scene.cell_offset = qtuple(cellOffset)
        self.map_scene.cell_diameter = cellDiameter

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

    def imageAtData(self, data):
        for imageItem in self.imageItems:
            if (imageItem.data(0) == data):
                break

        else:
            imageItem = None

        return imageItem

    def isWall(self, item):
        return (item in self.wallItems)
    
    def walls(self):
        return self.wallItems

    def isWallClosed(self, item):
        map_wall = item.data(0)
        return map_wall.closed

    def isWallHandle(self, item):
        return (item in self.wallHandleItems)

    def wallHandles(self):
        return self.wallHandleItems

    def isGridHandle(self, item):
        return (item == self.gridHandleItem)

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
        # setWallsVisible is called as part of updatefog, which is called when
        # moving the wall handles, don't do setVisible on them since they will
        # lose the focus and abort dragging, push them back instead (in
        # addition, not calling setVisible also fixes an infinite loop because
        # itemChanged reacts to visiblechanged, but this could be fixed ignoring
        # those for wall handles)

        # This is in the hot path of updatefog, don't do anything if already in
        # the right state, this prevents the busy work below and thus faster
        # updates when walls and wall handles are hidden
        if (self.allWallsItem.isVisible() == visible):
            return
        
        # setZValue generate itemChange on each, lock updates for those and
        # update at the end
        self.setLockDirtyCount(True)
        for handle in self.wallHandleItems:
            handle.setZValue(0.3 if visible else -1)
        self.setLockDirtyCount(False)
        self.makeDirty()
        self.allWallsItem.setVisible(visible)
    
    def setTokensVisible(self, visible, onlyHiddenFromPlayer = False):
        logger.info("visible %s onlyHiddenFromPlayer %s", visible, onlyHiddenFromPlayer)
        # self.makeDirty() 
        
        # XXX Don't update on every token change, do update at the end, note
        #     itemChange notifications are not blocked by blocksignals, either
        #     remove and restore the ItemSendsGeometryChanges flag for this item
        #     or roll some blocking counter
        for token in self.tokenItems:
            if ((not onlyHiddenFromPlayer) or token.data(0).hidden):
                # Hiding the token makes it lose focus, push it behind the image
                # instead
                # XXX Do the same thing for the other items? (walls, doors)
                if (visible):
                    token.setZValue(0.4)
                    
                else:
                    token.setZValue(-1)
                    
    def setDoorsVisible(self, visible):
        self.makeDirty()
        self.allDoorsItem.setVisible(visible)

    def adjustTokenGeometry(self, token):
        pixItem = self.getTokenPixmapItem(token)
        txtItem = self.getTokenLabelItem(pixItem)
        pix = pixItem.pixmap()
        map_token = pixItem.data(0)

        pixItem.setScale(map_token.scale / pix.width())
        # QGraphicsPixmapItem are not centered by default, do it
        pixItem.setOffset(-qSizeToPoint(pix.size() / 2.0))

        # Note 
        #   txtItem.document().setDefaultTextOption(QTextOption(Qt.AlignCenter))
        # does center the text in the width set with setTextWidth, but the item
        # anchor is still the item position so still needs to be set to the top
        # left corner, which changes depending on the text length
        # Note setTextInteractionFlags(Qt.TextEditorInteraction) is set on
        # demand when editing the label
        txtItem.setScale(1.0/pixItem.scale())
        # Calculate the position taking into account the text item reverse scale
        pos = QPointF(
            0 - txtItem.boundingRect().width() / (2.0 * pixItem.scale()), 
            pix.height() /2.0 - txtItem.boundingRect().height() / (2.0 * pixItem.scale())
        )
        txtItem.setPos(pos)

        # XXX This should probably have an array of labels (corners, center) or
        #     icons/badges
        txtItem = token.childItems()[1]
        center = getattr(map_token, "center", "")
        if (center == ""):
            txtItem.setHtml("")

        else:
            txtItem.setHtml("<div style='font-weight:bold; background:rgb(255, 255, 255, 64);'>%s</div>" % center)
        txtItem.setScale(1.0/pixItem.scale())
        pos = QPointF(
            0 - txtItem.boundingRect().width() / (2.0 * pixItem.scale()), 
            0 - txtItem.boundingRect().height() / (2.0 * pixItem.scale()),
        )
        txtItem.setPos(pos)
        
        # XXX This is not really token geometry, but it centralizes the tooltip
        #     update when the label is updated or when any other stat is updated
        #     in the CombatTracker, etc
        pixItem.setToolTip("<b>%s</b><br><i>%s</i><br><b>T0</b>:%s <b>AT</b>:%s <b>D</b>:%s<br><b>AC</b>:%s <b>HP</b>:%s" %(
            map_token.name,
            map_token.ruleset_info.Id,
            map_token.ruleset_info.T0,
            map_token.ruleset_info.AT,
            map_token.ruleset_info.Damage,
            map_token.ruleset_info.AC,
            map_token.ruleset_info.HP if (map_token.ruleset_info.HP_ == "") else map_token.ruleset_info.HP_
        ))
        
    def setTokenLabelText(self, token, text):
        logger.debug("%s", text)
        txtItem = self.getTokenLabelItem(token)
        # Use HTML since it allows setting the background color
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
        pixItem.setPos(*map_token.scene_pos)
        pixItem.setData(0, map_token)
        pixItem.setCursor(Qt.SizeAllCursor)
        # Note setting parent zvalue is enough, no need to set txtItem
        pixItem.setZValue(0.4)
        self.setTokenHiddenFromPlayer(pixItem, map_token.hidden)

        txtItem = QGraphicsTextItem(pixItem)
        self.setTokenLabelText(pixItem, map_token.name)
        # Keep the label always at the same size disregarding the token size,
        # because the label is a child of the pixitem it gets affected by it.
        # XXX This font size needs to be scaled by map size otherwise it looks
        #     small on big maps? 
        # Also reduce the font a bit
        font = txtItem.font()
        font.setPointSize(txtItem.font().pointSize() *0.75)
        txtItem.setFont(font)
        # No need to set TextTokenInteraction since it's disabled by default and
        # enabled on demand

        centerItem = QGraphicsTextItem(pixItem)
        font = txtItem.font()
        font.setPointSize(font.pointSize() * 1.5)
        centerItem.setFont(font)

        # XXX Have corner badges or status icons? 
        #     See https://game-icons.net/ https://github.com/game-icons/icons for status icons
        # XXX Fade token / draw cross on HP_ < 0?

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

    def addHandleItem(self, point, pen, data0 = None):
        handleDiameter = self.cellDiameter * 0.1
        # QGraphicsRectItem handles position and the rect topleft independently,
        # use the topleft to offset the rect so the position matches the rect's
        # center
        handleItem = GraphicsRectNotifyItem(-handleDiameter*0.5, -handleDiameter*0.5, handleDiameter, handleDiameter)
        handleItem.setPos(point)
        handleItem.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsScenePositionChanges)
        handleItem.setPen(pen)
        handleItem.setZValue(0.3)
        handleItem.setCursor(Qt.SizeAllCursor)
        handleItem.setData(0, data0)
        self.addItem(handleItem)
        
        return handleItem

    def addWall(self, map_wall):
        # XXX This is high frequency on caverns because of spurious setscene calls
        assert None is logger.debug("Adding wall %s", map_wall)
        pen = QPen(Qt.cyan)

        # Add a path and duplicate the first point if closed
        qpp = QPainterPath()
        # XXX Should use GraphicsPolygonItem if closed and GraphicsPathItem
        #     otherwise? (but it's not clear it's much faster and it creates
        #     another dragging codepath)
        # XXX Using GraphicsPathItem requires modifying the whole path when
        #     dragging, use independent lines? Create a polyline graphicsitem?
        qpoints = [QPointF(*p) for p in map_wall.points]
        # Don't modify qpoints, it's used below to add handles and don't need
        # to create handles for the dummy point closing a closed polygon
        if (map_wall.closed):
            qpp.addPolygon(QPolygonF(qpoints + qpoints[0:1]))
        else:
            qpp.addPolygon(QPolygonF(qpoints))
        wallItem = QGraphicsPathItem(qpp)
        # Items cannot be selected, moved or focused while inside a group, the
        # group can be selected and focused but not moved
        ##item.setFlags(QGraphicsItem.ItemIsFocusable | QGraphicsItem.ItemIsSelectable  | QGraphicsItem.ItemIsMovable)
        wallItem.setPen(pen)
        wallItem.setData(0, map_wall)
        self.wallItems.add(wallItem)
        self.allWallsItem.addToGroup(wallItem)

        for i, point in enumerate(qpoints):
            handleItem = self.addHandleItem(point, pen, wallItem)
            handleItem.setData(1, i)
            self.wallHandleItems.add(handleItem)

    def removeWall(self, wallItem):
        logger.info("wall %s", wallItem)
        self.wallItems.remove(wallItem)
        self.allWallsItem.removeFromGroup(wallItem)
        # Note removeFromGroup reparents the item to the scene (per docs), now
        # it's necessary to remove it from the scene
        self.removeItem(wallItem)

        # Remove from the set by safely iterating over a list copy 
        for handleItem in list(self.wallHandleItems):
            if (self.getWallItemFromWallHandleItem(handleItem) == wallItem):
                self.wallHandleItems.remove(handleItem)
                self.removeItem(handleItem)

    def addCircularWall(self, scenePos, sides = 4):
        # XXX Refactor this for light range, but light range is inside
        #     compute_fog which doesn't know about gscene?
        r = self.getCellDiameter() * 1.0
        
        points = []
        for s in xrange(sides):
            x, y = (
                scenePos.x() + r*math.cos(2.0 * s * math.pi / sides), 
                scenePos.y() + r*math.sin(2.0 * s * math.pi / sides)
            )
            points.append([x, y])
        
        wall = Struct(points=points, closed=True)

        self.map_scene.map_walls.append(wall)
        self.addWall(wall)

        self.makeDirty()

    def getWallItemFromWallHandleItem(self, handleItem):
        return handleItem.data(0)

    def getPointIndexFromWallHandleItem(self, handleItem):
        # Note storing lists into setData causes the list to be copied (verified
        # by checking the Python ID), so this needs to store the index into the
        # point list and then retrieve the point from the list so any updates
        # (done by the graphicsscene and elswhere) are noticed
        return handleItem.data(1)
        
    def addDoor(self, map_door):
        logger.info("Adding door %s", map_door.points)
        pen = QPen(Qt.black)
        
        # Doors are implicitly closed polylines (ie polygons without the last
        # point duplicated), duplicate the last point and create as polygons
        qpoints = [QPointF(*point) for point in map_door.points]
        qpoints.append(qpoints[0])
        
        item = QGraphicsPolygonItem(QPolygonF(qpoints))
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

    def addGrid(self):
        rect = self.itemsBoundingRect()
        
        pen = QPen(QColor(0, 0, 0, 128))
        x, y = self.cellAnchor.x(), self.cellAnchor.y()
        while (x < rect.right()):
            lineItem = QGraphicsLineItem(x, rect.top(), x, rect.bottom())
            lineItem.setPen(pen)
            self.gridLineItems.add(lineItem)
            self.gridItem.addToGroup(lineItem)
            x += self.cellDiameter

        while (y < rect.bottom()):
            lineItem = QGraphicsLineItem(rect.left(), y, rect.right(), y)
            lineItem.setPen(pen)
            self.gridLineItems.add(lineItem)
            self.gridItem.addToGroup(lineItem)
            y += self.cellDiameter

        # XXX This should be in some global place
        # XXX The position needs to be always updated even if already created?
        if (self.gridHandleItem is None):
            # Add handles to pan and size the grid
            # QGraphicsRectItem handles position and the rect topleft
            # independently, use the topleft to offset the rect so the position
            # matches the rect's center
            # XXX This should be hidden on setGridVisible?
            handleItem = self.addHandleItem(self.cellAnchor + QPointF(self.cellDiameter, self.cellDiameter), Qt.red,self.gridItem)
            self.gridHandleItem = handleItem

    def removeGrid(self):
        logger.info("")
        
        # Remove the lines but not the grid item
        # XXX Is there a way to iterate through group items without having to
        #     keep them externally?
        for lineItem in self.gridLineItems:
            self.gridItem.removeFromGroup(lineItem)
            # Note removeFromGroup reparents the item to the scene (per docs),
            # now it's necessary to remove it from the scene
            self.removeItem(lineItem)
            
        self.gridLineItems.clear()
        
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

    def getLightRanges(self):
        # Light ranges in feet, no need to sort these, they get sorted below
        d = [ 
            ("None", 0.0), ("candle", 5.0), ("torch", 15.0), 
            ("light spell", 20.0), ("lantern", 30.0), ("campfire", 35.0),
            ("infravision", 60.0),
        ]
        # Return light ranges in scene units
        # XXX Assumes 5ft per cell, lightRange uses scene units
        lightRanges = [(i[0], i[1] * self.getCellDiameter() / 5.0) for i in sorted(d, cmp= lambda a,b : cmp(a[1], b[1]))]

        return lightRanges

    def getLightRangeName(self, lightRange):
        lightRanges = self.getLightRanges()
        i = index_of([l[1] for l in lightRanges], lightRange)
        return lightRanges[i][0]
        
    def setFogVisible(self, visible):
        logger.info("setFogVisible %s", visible)
        self.fogVisible = visible
        self.updateFog()

    def setBlendFog(self, blend):
        logger.info("setBlendFog %s", blend)
        self.blendFog = blend
        self.updateFog()
        
    def setFogColor(self, color):
        self.fogColor = color
        # XXX This should update the fog if done after initialization

    def updateFog(self, force=False):
        logger.info("force %s dirty %s draw %s blend %s", force, self.fogPolysDirtyCount != self.dirtyCount, self.fogVisible, self.blendFog)
        gscene = self
        
        if (gscene.getFogCenter() is None):
            logger.warning("Called updateFog with no token item!!")
            return

        if (self.blendFog):
            fogBrush = QBrush(QColor(0, 0, 255, 125))
            fogPen = QPen(Qt.black)

        else:
            # Match the background color to prevent leaking the map dimensions
            fogBrush = QBrush(self.fogColor)
            fogPen = QPen(self.fogColor)
        
        # XXX Use gscene.blockSignals to prevent infinite looping 
        #     and/or updateEnabled(False) to optimize redraws
        if ((gscene.dirtyCount != gscene.fogPolysDirtyCount) or force):
            gscene.fogPolysDirtyCount = gscene.dirtyCount
            
            token_pos = self.getFogCenter()
            token_x, token_y = qtuple(token_pos)
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
    # XXX Investigate 
    #       token_x, token_y = 918, 648 
    #       wall = 910.8699293051828, 648.5932296062014, 920.3038924799687, 641.7909349343238
    #     The token is under the wall but the fog is calculated as if above?
    logger.info("draw fog %d polys pos %s", len(scene.map_walls), fog_center)
    
    token_x, token_y = fog_center
    fog_polys = []

    # Place a fake circle wall to simulate light range
    # XXX Note the fake circle wall doesn't account for other lights in the 
    #     scene beyond this range
    points = []
    if (light_range != 0.0):
        r = light_range * 1.0
        sides = 16
        for s in xrange(sides):
            x, y = (
                token_x + r*math.cos(2.0 * s * math.pi / sides), 
                token_y + r*math.sin(2.0 * s * math.pi / sides)
            )
            points.append([x, y])
    light_wall = Struct(points=points, closed=True)

    light_range2 = light_range ** 2
    
    # Include the light rectangle centered in the token since it can place walls
    # outside the current bounds, otherwise the bound clipping would interfere
    # with the fake circle wall and invert the fog
    
    # XXX Probably something more efficient could be done like clipping the wall
    #     against the bounds, or this can be removed once lightranges are
    #     implemented by clearing the fog mask. In addition, this causes stale
    #     rendering artifacts when using the "blend fog" debugging option since
    #     the graphicsscene doesn't expect drawing outside of the sceneRect. 
    epsilon = 10.0
    bounds = bounds.united(
        QRectF(token_x - light_range - epsilon, token_y - light_range - epsilon, 
            (light_range + epsilon)*2, (light_range + epsilon)*2))

    max_min_l = (bounds.width() ** 2 + bounds.height() ** 2)
    for map_item in scene.map_walls + scene.map_doors + [light_wall]:
        # XXX Make doors be special walls?
        is_door = hasattr(map_item, "open")
        is_wall = hasattr(map_item, "closed")
        if (is_door and map_item.open):
            continue

        points = map_item.points
        # Duplicate first point in last point for doors and closed polylines
        if (is_door or map_item.closed):
            # Note this duplication is safe even if there are no points
            points = points + points[0:1]

        for i in xrange(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            
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
            # (slow) and which doesn't work anyway for degenerate cases like
            # very close to a wall where the frustum is close to 180 degrees
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
                # XXX All this could be put into a table
                
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

    def detectAndGenerateWallAtPos(self, pos):
        """
        Steps:
        1. Detect the maximum square that fits inside the wall (wall depth)
        2. Use the wall depth to drive the downscale and erode filter sizes
           below
        3. Collect path positions via flood fill with downsize and erosion in
           order to remove fine features
        4. Thin the wall to the skeleton points
        5. Build pixel-long polylines from the skeleton points
        6. Merge multiple straight-ish pixel-long polylines into a single
           straight line, with some tolerance

        Downscaling serves two purposes:
        - Performance, process less pixels, especially necessary for edge
          thinning which is performed multiple times since only two pixels of
          width are removed per iteration
        - Quantize, remove too much detail, less different coordinates and
          straighter lines (but the hough space has its own additional
          quantization)

        The erosion filter diameter is necessary to remove fine features so they
        are not contoured (eg hatching, grids), but it also helps with
        performance since it returns a thinner contour and thinning is a slow
        iterative process (not clear where the perf breaking point between
        erosion kernel size and number of thinning iterations is)

        Ideally there would be no downscaling, the erosion filter would remove
        all fine features, but performance suffers. Also downscaling performs
        important quantization so contiguous points can be merged into a
        straight line

        References

        - https://note.nkmk.me/en/python-numpy-image-processing/
        - https://note.nkmk.me/en/python-numpy-opencv-image-binarization/
        - https://msameeruddin.hashnode.dev/image-dilation-explained-in-depth-using-numpy
        - https://msameeruddin.hashnode.dev/image-erosion-explained-in-depth-using-numpy
        - https://tempflip.medium.com/lane-detection-with-numpy-2-hough-transform-f4c017f4da39
        - https://alyssaq.github.io/2014/understanding-hough-transform/
        - https://github.com/okaneco/skeletonize#reference
        - https://dl.acm.org/doi/10.1145/357994.358023
        - https://www.csie.ntu.edu.tw/~hil/paper/cacm86.pdf
        - https://dl.acm.org/doi/10.1145/321637.321646
        - https://iris.unimore.it/retrieve/e31e124e-f5e5-987f-e053-3705fe0a095a/2019_ICIAP_Improving_the_Performance_of_Thinning_Algorithms_with_Directed_Rooted_Acyclic_Graphs.pdf
        - https://homepages.inf.ed.ac.uk/rbf/HIPR2/wksheets.htm
        """
        
        # XXX Find out why hough space theta quantization doesn't seem to help
        #     quantizing 

        # XXX Do downscale after erosion for more quantization?
        
        # XXX Do something similar to help finding the ideal grid size on maps
        #     with grids (even detect multiple grid cells and average), use a
        #     horizontal/vertical or corner filter

        # XXX Do some progress dialog box, see 
        #     https://stackoverflow.com/questions/47879413/pyqt-qprogressdialog-displays-as-an-empty-white-window
        
        gscene = self.scene()

        # XXX This hardcodes the image to autodetect to 0
        imageItem = gscene.imageAt(0)
        pixmap = imageItem.pixmap()
        a = np_pixmaptoarray(pixmap)
        a_bak = a

        w, h = pixmap.width(), pixmap.height()

        pixel = imageItem.mapFromScene(self.mapToScene(pos))
        x, y = qtuple(pixel.toPoint())

        # Find the wall size that will drive the downscale factor
        # and the erosion kernel size
        
        # Full-image grayscale and thresholding can take long on big maps, this
        # is done inside np_findmaxfiltersize at per erode-size level
        k = np_findmaxfiltersize(a, x, y, 64)

        # There's no hard logic to this other making 
        #       downscale + filter_diameter <= k 
        # and empirically testing values that look ok
        downscale = int(math.log(k, 2)) - 1
        filter_diameter = int(math.log(k, 2))
        
        while (True):
            logger.info("found max_k %d downscale %d filter_diameter %d", k, downscale, filter_diameter)

            QApplication.setOverrideCursor(Qt.WaitCursor)
            # This is necessary so the mouse pointer change is made visible
            qApp.processEvents()

            pixmap2 = pixmap
            if (downscale > 1):
                logger.info("Downscaling %d", downscale)
                pixmap2 = pixmap.scaledToWidth(pixmap.width() / downscale)
                if (save_np_debug_images):
                    logger.info("Saving downscale")
                    pixmap2.save(os.path.join("_out", "downscaled.png"))

            # Recalculate since the image may have been downscaled
            pixel = imageItem.mapFromScene(self.mapToScene(pos)) / downscale
            x, y = qtuple(pixel.toPoint())

            a = np_pixmaptoarray(pixmap2)
            
            # XXX Ideally grayscale/hue and threshold value should be another
            #     search parameter/extracted from pixel?
            # XXX Grayscale and threshold could be done just in time?
            a = np_graythreshold(a, 64)
            active = np_floodfillerode(a, x, y, filter_diameter)
            b = a_bak.copy()
            if (len(active) > 0):
                j, i = np.array(list(active)).T * downscale
                for jj in xrange(downscale):
                    for ii in xrange(downscale):
                        # Note each write is a broadcast to all elements in the
                        # set so it cannot be converted into a single index
                        # range write (the other option is to use an index range
                        # write inside a set element loop, but the number of
                        # elements in the set is normally a lot larger than
                        # downscale, so probably slower)
                        b[j + jj, i + ii] = (128, 0, 255, 255)
                
            qimage = QImage(b.data, b.shape[1], b.shape[0], QImage.Format_RGB32)
            pixmap3 = QPixmap.fromImage(qimage)
            imageItem.setPixmap(pixmap3)

            qApp.restoreOverrideCursor()

            # XXX This should allow scrolling the VTTGraphicsView in that mode,
            #     but then the dialog needs to be non-modal?
            text, ok = QInputDialog.getText(
                # XXX This needs to be the main window or the buttons will fail
                #     to lose focus and won't work when clicked (could be due to
                #     findParentDock returning the dock parent and changing the
                #     style if using VTTGraphicsView instead of VTTMainWindow).
                #     Special case findParentDock to ignore QInputDialog or
                #     abstract it out instead of hard-coding to parent.parent?
                self.parent().parent(),
                "Autowall Parameters (%dx%d %d)" % (w, h, k),
                "downscale, filter size:", QLineEdit.Normal, 
                "%d, %d" % (downscale, filter_diameter)
            )
            if ((not ok) or (text == "")):
                # XXX This replicates cleanup, set a flag and use a common path
                imageItem.setPixmap(pixmap)
                return
            new_downscale, new_filter_diameter = [int(i) for i in text.split(",")]
            
            if ((new_downscale, new_filter_diameter) == (downscale, filter_diameter)):
                break
            downscale, filter_diameter = new_downscale, new_filter_diameter

        imageItem.setPixmap(pixmap)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        # This is necessary so the mouse pointer change is made visible
        qApp.processEvents()
        
        active = np_findcontourpoints(a, active)
        polylines, tjunctions = np_connectcontourpoints(a, active)
        max_coalesce_dist2 = (gscene.map_scene.cell_diameter*downscale / 8.0) ** 2
        walls = np_coalescepoints(a, polylines, tjunctions, max_coalesce_dist2)

        # XXX Have options to grid snap the walls?
        # XXX Have options to do extra downscale?

        # This can be many walls, block scene updates
        # XXX Not clear this is really blocking emits?
        gscene.setLockDirtyCount(True)
        for points in walls:
            wall = Struct(points=[qtuple(imageItem.mapToScene(QPointF(*p)*downscale)) for p in points], closed=False)
            gscene.map_scene.map_walls.append(wall)
            gscene.addWall(wall)
        gscene.setLockDirtyCount(False)
        gscene.makeDirty()
        gscene.invalidate()
            
        # XXX This needs a context guard to protect against leaving the wrong
        #     cursor in the presence of exceptions
        qApp.restoreOverrideCursor()
    
    def mousePressEvent(self, event):
        logger.info("%s", EventTypeString(event))

        gscene = self.scene()

        focusItem = gscene.focusItem()
        # XXX This needs to check there's no item under the mouse already?
        if ((focusItem is None) and (len(gscene.selectedItems()) == 1)):
            # When panning the view by dragging, the focus gets lost but the 
            # selection stays in the old focused item, use the selected item instead
            # XXX In general use the selected item instead of the focused one?
            focusItem = gscene.selectedItems()[0]
        if (int(event.modifiers() & Qt.ControlModifier) != 0):
            # XXX This should set the Qt.CrossCursor when ctrl is pressed and
            #     mouse is moved?
            if (gscene.isGridHandle(focusItem)):
                pass

            elif (gscene.isWallHandle(focusItem)):
                # Add a point to the focused wall on ctrl + click
                
                # Add the point after the focused one unless the focused one is the
                # first one of an open wall, in which case add it before the focused
                # one (this allows growing open walls at the start or at the end)

                # XXX When inserting a point between two existing points, this
                #     could add to the segment that is closest instead of always
                #     adding between the focused and the next point
                wallHandle = focusItem
                wallItem = gscene.getWallItemFromWallHandleItem(wallHandle)
                i = gscene.getPointIndexFromWallHandleItem(wallHandle)
                map_wall = wallItem.data(0)
                # On open walls always extend the extremes, otherwise extend the
                # next point to the currently focused handle
                if (map_wall.closed or (i > 0)):
                    i += 1
                logger.info("Inserting wall point at %d", i)
                # Don't snap the pos since it will cause the new handle to be
                # away from the mouse pointer and abort the dragging
                scenePos = self.mapToScene(event.pos())
                map_wall.points.insert(i, qlist(scenePos))

                # XXX This needs a new wall handle and update the painterpath,
                #     ideally would just setscene for the heavy-handed case but
                #     can't be called from inside VTTGraphicsView, remove and add
                #     the wall instead
                gscene.removeWall(wallItem)
                gscene.addWall(map_wall)


            else:
                # Create a one-point wall
                
                # XXX Is a one-point wall problematic? this could create a two
                #     point wall but it's counter intuitive?
                
                # XXX This needs a new wall handle and update the painterpath,
                #     ideally would just setscene for the heavy-handed case but
                #     can't be called from inside VTTGraphicsView, remove and add
                #     the wall instead
                scenePos = self.mapToScene(event.pos())
                # XXX This could have a modifier to select between open and closed
                #     or detect close at end via double click or start==end?
                map_wall = Struct(points=[qlist(scenePos)], closed=False)
                gscene.map_scene.map_walls.append(map_wall)
                gscene.addWall(map_wall)
                
            # No need to focus the new point since it will be focused by
            # calling super below
            gscene.makeDirty()
                

        super(VTTGraphicsView, self).mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        logger.info("%s", EventTypeString(event))

        # Call the inherited event handler to update the focused item before
        # checking what the focused item is below
        super(VTTGraphicsView, self).mouseDoubleClickEvent(event)

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
            
            elif (gscene.isGridHandle(focusItem)):
                self.queryGridParams()

        else:
            self.detectAndGenerateWallAtPos(event.pos())

    def queryGridParams(self):
        # XXX This should autodetect the grid restricted to the first
        #     cell, pasting the parameters to the dialog box
        gscene = self.scene()
        text, ok = QInputDialog.getText(
            self,
            "Grid Parameters", 
            "Grid cell (offsetx, offsety, diameter):", QLineEdit.Normal, 
            "%d, %d, %d" % (gscene.cellOffset.x(), gscene.cellOffset.y(), gscene.cellDiameter)
        )
        if ((not ok) or (text == "")):
            return

        cellOffsetX, cellOffsetY, cellDiameter = [float(i) for i in text.split(",")]
        cellOffset = QPointF(cellOffsetX, cellOffsetY)
        gscene.setCellOffsetAndDiameter(cellOffset, cellDiameter)
        gscene.gridHandleItem.setPos(gscene.cellAnchor + QPointF(cellDiameter, cellDiameter))
        
        # Remove and add the grid back to refresh it
        gscene.removeGrid()
        # Dirty the scene (cell diameter or offset changed, so any item
        # that depends on that should change) and the grid
        gscene.makeDirty()
        gscene.addGrid()


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
        
        # Prevent propagation
        event.accept()

    def event(self, event):
        logger.info("type %s", EventTypeString(event.type()))
        # Tab key needs to be trapped at the event level since it doesn't get to
        # keyPressEvent
        if ((event.type() == QEvent.KeyPress) and (event.key() in [Qt.Key_Tab, Qt.Key_Backtab]) and 
            (int(event.modifiers() & Qt.ControlModifier) == 0)):
            gscene = self.scene()

            focusItem = gscene.focusItem()

            # Tab through wall handles if the focused item is a wall handle,
            # otherwise tab through tokens
            if (gscene.isWallHandle(focusItem) or (len(gscene.tokens()) == 0)):
                # XXX sets have no order, this is not very useful without some
                #     order since it goes to some random handle, not to the 
                #     closest one
                items = list(gscene.wallHandles())

            else:
                items = list(gscene.tokens())

            
            itemCount = len(items)
            if (itemCount > 0):
                delta = 1 if (event.key() == Qt.Key_Tab) else -1
                # XXX Note token_items is a set, so it doesn't preserve the
                #     order, may not be important since it doesn't matter
                #     there's no strict order between tokens as long as it's
                #     consistent one (ie the order doesn't change between tab
                #     presses as long as no items were added to the token set)
                # XXX Should probably override GraphicsView.focusNextPrevChild
                #     once moving away from filtering?
                
                focusedIndex = index_of(items, gscene.focusItem())
                if (focusedIndex == -1):
                    # Focus the first or last item
                    focusedIndex = itemCount
                else:
                    # Clear the selection rectangle on the old focused item
                    items[focusedIndex].setSelected(False)

                focusedIndex = (focusedIndex + delta) % itemCount
                focusedItem = items[focusedIndex]
                
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
        focusItem = gscene.focusItem()
        if ((gscene.isToken(focusItem) or gscene.isWallHandle(focusItem) 
            or gscene.isGridHandle(focusItem)) and 
            (event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down])):
            d = { Qt.Key_Left : (-1, 0), Qt.Key_Right : (1, 0), Qt.Key_Up : (0, -1), Qt.Key_Down : (0, 1)}
            # Snap to half cell and move in half cell increments
            # XXX Make this configurable
            # XXX Keep track of the "heading" (even if there's a collision),
            #     could even be shown on the token and will with which door to
            #     open/close when there are multiple adjacent doors
            # XXX Tokens don't move if snap is disabled, investigate?
            if (self.snapToGrid):
                snap_granularity = gscene.getCellDiameter() / 2.0

            else:
                snap_granularity = 1.0

            if (int(event.modifiers() & Qt.ShiftModifier) != 0):
                snap_granularity /= 10.0
            move_granularity = snap_granularity
                
            delta = QPointF(*d[event.key()]) * move_granularity

            # Snap in case it wasn't snapped before, this will also allow using
            # the arrow keys to snap to the current cell if the movement is
            # rejected below
            if (gscene.isGridHandle(focusItem)):
                snapPos = focusItem.pos()

            else:
                snapPos = gscene.snapPositionWithCurrentSettings(focusItem.pos())
            
            # Note when the token is a group, the children are the one grabbed,
            # not the group, use the focusitem which is always the group

            if (gscene.isWallHandle(focusItem)):
                # Update the handle and the adjacent walls
                gscene.setWallHandleItemPos(focusItem, snapPos + delta)

            elif (gscene.isGridHandle(focusItem)):
                # Update the grid handle, will cascade into an update of the grid
                
                # XXX This doesn't work for ctrl+left/right keys which are
                #     swallowed by the music player, fix? (but the x delta
                #     cannot be easily applied anyway for both mouse and
                #     keyboard homogeneously anyway?)
                logger.info("setting pos %s", snapPos + delta)
                focusItem.setPos(snapPos + delta)

            else:
                # Intersect the path against the existing walls and doors, abort
                # the movement if it crosses one of those

                # Use a bit of slack to avoid getting stuck in the intersection
                # point due to floating point precision issues, don't use too
                # much (eg. 1.5 or more) or it will prevent from walking through
                # tight caves
                
                # XXX Note this doesn't take the size into account, can get
                #     tricky for variable size tokens, since proper intersection
                #     calculation requires capsule-line intersections
                
                # XXX This doesn't guarantee that there's a given separation
                #     between token and wall, eg by moving the token vertically,
                #     it could start moving parallel to a vertical wall that is
                #     just at the token position (since moving vertically won't
                #     collide against vertical walls unless they are completely
                #     edge on). Needs to check distance to line or similar?
                l = QLineF(snapPos, snapPos + delta * 1.25)

                # Note walls are lines, not polys so need to check line to line
                # intersection instead of poly to line. Even if that wasn't the
                # case, PolygonF.intersect is new in Qt 5.10 which is not
                # available on this installation, so do only line to line
                # intersection
                
                # XXX Convert to a single path for all the walls and do a single
                #     line vs. path intersection (but path intersect docs say
                #     "non closed paths will be treated as implicitly closed"?
                #     also check collidesWithPath?
                i = QLineF.NoIntersection
                for wall_item in gscene.walls():
                    # XXX This should convert line to path and intersect paths
                    # XXX Get the walls from the scene not the gscene?
                    path = wall_item.path()
                    for j in xrange(path.elementCount() - (0 if gscene.isWallClosed(wall_item) else 1)):
                        element1 = path.elementAt(j)
                        element2 = path.elementAt((j+1) % path.elementCount())
                        assert element1.isLineTo() or ((j == 0) and element1.isMoveTo())
                        assert element2.isLineTo() or (((j+1) == path.elementCount()) and element2.isMoveTo())
                        ll = QLineF(element1.x, element1.y, element2.x, element2.y)
                        i = l.intersect(ll, None)
                        logger.debug("wall intersection %s %s is %s", qtuple(l), qtuple(ll), i)
                        
                        if (i == QLineF.BoundedIntersection):
                            logger.debug("Aborting token movement, found wall intersection %s between %s and %s", i, qtuple(l), qtuple(ll))
                            break
                    if (i == QLineF.BoundedIntersection):
                        break
                    
                else:
                    # Check closed doors
                    # XXX intersects is not on this version of Qt (5.10), roll
                    #     manual checks for now, but change to path intersection
                    #     at some point?
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
                        # No collision, allow the position
                        focusItem.setPos(snapPos + delta)
                        
            self.ensureVisible(focusItem, self.width()/4.0, self.height()/4.0)
            # Note programmatic setpos doesn't trigger itemchange, dirty the 
            # scene so the fog polygons are recalculated on the next updateFog
            # call.
            gscene.makeDirty()

        elif (gscene.isToken(focusItem) and (event.key() == Qt.Key_Return)):
            # Enter editing mode on the token label (no need to check for
            # txtItem not focused since the token is focused and the scene focus
            # textitem and token independently)
            # XXX This is replicated across doubleclick, refactor
            # XXX This should select the whole text
            # XXX Should this use the widget .setFocus  or the scene setfocusItem/setSelected?
            txtItem = gscene.getTokenLabelItem(focusItem)
            txtItem.setTextInteractionFlags(Qt.TextEditorInteraction)
            txtItem.setFocus(Qt.TabFocusReason)
    
        elif (gscene.isGridHandle(focusItem) and (event.key() == Qt.Key_Return)):
            self.queryGridParams()

        elif (gscene.isToken(focusItem) and (event.text() in ["-", "+"])):
            # Increase/decrease token size, no need to recenter the token on the
            # current position since tokens are always centered on 0,0
            delta = 1 if event.text() == "+" else -1
            map_token = focusItem.data(0)
            deltaScale = delta * (gscene.getCellDiameter() / 4.0)
            map_token.scale += deltaScale

            gscene.adjustTokenGeometry(focusItem)

            gscene.makeDirty()

        elif (gscene.isToken(focusItem) and (event.text() == " ")):
            # Open the adjacent door
            # XXX Do something about when two doors are adjacent, have a player
            #     "heading" 
            threshold2 = (gscene.getCellDiameter() ** 2.0) * 1.1
            token_center = focusItem.sceneBoundingRect().center()
            for door_item in gscene.doors():
                door_center = door_item.sceneBoundingRect().center()
                v = (door_center - token_center)
                dist2 = QPointF.dotProduct(v, v)
                logger.info("checking token %s vs. door %s %s vs. %s", token_center, door_center, dist2, threshold2)
                if (dist2 < threshold2):
                    gscene.setDoorOpen(door_item, not gscene.isDoorOpen(door_item))
                    gscene.makeDirty()
                    break

        elif (gscene.isToken(focusItem) and (event.text() == "h")):
            # Hide token from player view
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
                delta = 1 if event.text() == "l" else -1
                lightRanges = gscene.getLightRanges()
                lightRangeIndex = index_of([l[1] for l in lightRanges], gscene.getLightRange())
                lightRangeIndex = (lightRangeIndex + delta + len(lightRanges)) % len(lightRanges)
                lightRange = lightRanges[lightRangeIndex][1]
                gscene.setLightRange(lightRange)
                # XXX This needs to update the status bar
                
            elif (event.text() == "s"):
                # Toggle snap to grid
                self.snapToGrid = not self.snapToGrid
                gscene.setSnapToGrid(self.snapToGrid)
                # XXX This needs to update the status bar
                
            elif (event.text() == "w"):
                # Toggle wall visibility on DM View
                # XXX Cycle through wall+handles visible, wall visible, both
                #     invisible
                self.drawWalls = not self.drawWalls
                # Always draw doors otherwise can't interact with them
                # XXX Have another option? Paint them transparent?
                # XXX Is painting transparent faster than showing and hiding?
                gscene.setWallsVisible(self.drawWalls)

            else:
                super(VTTGraphicsView, self).keyPressEvent(event)
            
        else:
            super(VTTGraphicsView, self).keyPressEvent(event)

            
    def eventFilter(self, source, event):
        logger.debug("source %s type %s", class_name(source), EventTypeString(event.type()))

        return super(VTTGraphicsView, self).eventFilter(source, event)


class VTTMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(VTTMainWindow, self).__init__(parent)

        self.profiling = False

        self.campaign_filepath = None
        self.recent_filepaths = []

        self.gscene = None
        self.scene = None

        self.fogColor = QColor(0, 0, 0)

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
        # Set the background color to prevent leaking the map dimensions
        imageWidget.setBackgroundColor(self.fogColor)
        # Allow the widget to receive keyboard events
        imageWidget.setFocusPolicy(Qt.StrongFocus)
        imageWidget.installEventFilter(self)
        self.imageWidget = imageWidget

        view = VTTGraphicsView()
        self.graphicsView = view
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        
        tree = QTreeWidget()
        self.tree = tree
        # Don't focus this on wheel scroll
        tree.setFocusPolicy(Qt.StrongFocus)
        tree.itemChanged.connect(self.treeItemChanged)
        tree.itemActivated.connect(self.treeItemActivated)


        self.docEditor = DocEditor()
        textEditor = self.docEditor.textEdit
        self.textEditor = textEditor

        def cursorPositionChanged(editor):
            logger.info("")

            hLevel = editor.textCursor().charFormat().property(QTextFormat.FontSizeAdjustment)
            if (hLevel is not None):
                hLevel = "<h%d>" % (3 - hLevel + 1)
            else:
                hLevel = "<p>"

            hasRuler = (editor.textCursor().blockFormat().property(QTextFormat.BlockTrailingHorizontalRulerWidth) is not None)
            # XXX Missing some stats? (chars, blocks, bytes, images, tables, sizes)
            # XXX Missing indent
            s = "%s %d %s%s%s%s%s %s" % (
                editor.fontFamily(), editor.fontPointSize(), hLevel,
                " <b>" if (editor.fontWeight() == QFont.Bold) else "",
                " <i>" if editor.fontItalic() else "", 
                " <u>" if editor.fontUnderline() else "", 
                " <hr>" if hasRuler else "",
                qAlignmentFlagToString(editor.alignment())
            )
            # Prevent any char interpreted as HTML tags
            self.statusScene.setTextFormat(Qt.PlainText)
            # XXX Missing refreshing on format change
            self.statusScene.setText(s)
    
        def modifiedDocument(editor, changed):
            logger.info("")
            self.updateTextTitle(editor, changed)
            
        textEditor.document().modificationChanged.connect(lambda changed: modifiedDocument(textEditor, changed))
        textEditor.cursorPositionChanged.connect(lambda : cursorPositionChanged(textEditor))

        for view, title in [
            (self.tree, "Campaign"), 
            # XXX Once multiple scenes/rooms are supported, these two should be dynamic
            (self.imageWidget, "Player View - %s" % (g_server_ips,) ),
            (self.graphicsView, "DM View"),
        ]:
            self.wrapInDockWidget(view, title)

        self.setCentralWidget(self.docEditor)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)

        qApp.focusChanged.connect(self.focusChanged)

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
            if (className in ["ImageWidget", "QTreeWidget", "VTTGraphicsView", "DocEditor"]):
                if (className == "ImageWidget"):
                    self.imageWidget.parent().setObjectName(uuid)
                elif (className == "QTreeWidget"):
                    self.tree.parent().setObjectName(uuid)
                elif (className == "VTTGraphicsView"):
                    self.graphicsView.parent().setObjectName(uuid)
                elif (className == "DocEditor"):
                    self.docEditor.parent().setObjectName(uuid)

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

        # XXX Missing saving/restoring the textEditor
        
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
        self.openDysonUrlAct = QAction("Open D&yson URL...", self, shortcut="ctrl+shift+o", triggered=self.openDysonUrl)
        self.saveAct = QAction("&Save", self, shortcut="ctrl+s", triggered=self.saveScene)
        self.saveAsAct = QAction("Save &As...", self, shortcut="ctrl+shift+s", triggered=self.saveSceneAs)
        self.exitAct = QAction("E&xit", self, shortcut="alt+f4", triggered=self.close)
        self.profileAct = QAction("Profile", self, shortcut="ctrl+p", triggered=self.profile)

        self.recentFileActs = []
        for i in range(most_recently_used_max_count):
            self.recentFileActs.append(
                    QAction(self, visible=False, triggered=self.openRecentFile))

        
        self.cutItemAct = QAction("Cut& Item", self, shortcut="ctrl+x", triggered=self.cutItem)
        self.copyItemAct = QAction("&Copy Item", self, shortcut="ctrl+c", triggered=self.copyItem)
        # XXX Do ctrl+shift+v to paste text without format
        self.pasteItemAct = QAction("&Paste Item", self, shortcut="ctrl+v", triggered=self.pasteItem)
        self.deleteItemAct = QAction("&Delete Item", self, shortcut="del", triggered=self.deleteItem)
        self.deleteSingleItemAct = QAction("&Delete Single Item", self, shortcut="ctrl+del", triggered=self.deleteItem)
        self.importDungeonScrawlAct = QAction("Import &Dungeon Scrawl...", self, shortcut="ctrl+a", triggered=self.importDungeonScrawl)
        self.importTokenAct = QAction("Import &Token...", self, shortcut="ctrl+k", triggered=self.importToken)
        self.importImageAct = QAction("Import &Image...", self, shortcut="ctrl+i", triggered=self.importImage)
        self.importMusicAct = QAction("Import &Music Track...", self, shortcut="ctrl+m", triggered=self.importMusic)
        self.importHandoutAct = QAction("Import &Handout...", self, shortcut="ctrl+h", triggered=self.importHandout)
        self.importTextAct = QAction("Import Te&xt...", self, shortcut="ctrl+t", triggered=self.importText)
        self.clearWallsAct = QAction("Clear All &Walls...", self, triggered=self.clearWalls)
        self.clearDoorsAct = QAction("Clear All &Doors...", self, triggered=self.clearDoors)
        self.clearFogAct = QAction("Clear &Fog of War...", self, shortcut="ctrl+f", triggered=self.clearFog)
        self.copyScreenshotAct = QAction("DM &View &Screenshot", self, shortcut="ctrl+alt+c", triggered=self.copyScreenshot)
        self.copyFullScreenshotAct = QAction("DM &Full Screenshot", self, shortcut="ctrl+shift+c", triggered=self.copyFullScreenshot)
        self.newCombatTrackerAct = QAction("New Combat Trac&ker", self, triggered=self.newCombatTracker)
        self.newEncounterBuilderAct = QAction("New Encounter Buil&der", self, shortcut="ctrl+d", triggered=self.newEncounterBuilder)
        self.newBrowserAct = QAction("New &Browser", self, shortcut="ctrl+b", triggered=self.newBrowser)
        self.lockFogCenterAct = QAction("&Lock Fog Center", self, shortcut="ctrl+l", triggered=self.lockFogCenter, checkable=True)
        self.displayInitiativeOrderAct = QAction("&Display Initiative Order", self, triggered=self.displayInitiativeOrder, checkable=True)
        self.displayInitiativeOrderAct.setChecked(True)
        self.nextTrackAct = QAction("N&ext Music Track", self, shortcut="ctrl+e", triggered=self.nextTrack)
        self.rewindTrackAct = QAction("Rewind Music Track", self, shortcut="ctrl+left", triggered=self.rewindTrack)
        self.forwardTrackAct = QAction("Forward Music Track", self, shortcut="ctrl+right", triggered=self.forwardTrack)
        self.closeWindowAct = QAction("&Close Window", self, triggered=self.closeWindow)
        self.closeWindowAct.setShortcuts(["ctrl+f4", "ctrl+w"])
        self.prevWindowAct = QAction("&Previous Window", self, shortcut="ctrl+shift+tab", triggered=self.prevWindow)
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
        fileMenu.addAction(self.openDysonUrlAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.saveAct)
        fileMenu.addAction(self.saveAsAct)
        fileMenu.addSeparator()
        for action in self.recentFileActs:
            fileMenu.addAction(action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAct)
        fileMenu.addAction(self.profileAct)
        
        editMenu = QMenu("&Edit", self)
        editMenu.addAction(self.importDungeonScrawlAct)
        editMenu.addAction(self.importImageAct)
        editMenu.addAction(self.importTokenAct)
        editMenu.addAction(self.importMusicAct)
        editMenu.addAction(self.importHandoutAct)
        editMenu.addAction(self.importTextAct)
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
        editMenu.addAction(self.deleteSingleItemAct)
        editMenu.addSeparator()
        editMenu.addAction(self.clearDoorsAct)
        editMenu.addAction(self.clearWallsAct)
        editMenu.addAction(self.clearFogAct)
        editMenu.addSeparator()
        editMenu.addAction(self.copyScreenshotAct)
        editMenu.addAction(self.copyFullScreenshotAct)

        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.newBrowserAct)
        viewMenu.addAction(self.newCombatTrackerAct)
        viewMenu.addAction(self.newEncounterBuilderAct)
        viewMenu.addSeparator()
        viewMenu.addAction(self.lockFogCenterAct)
        viewMenu.addAction(self.displayInitiativeOrderAct)
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
                "<li> token collision"
                "<li> line of sight"
                "<li> fog of war"
                "<li> remote viewing via http"
                "<li> <a href=\"https://app.dungeonscrawl.com/\">Dungeon Scrawl</a> import"
                "<li> combat tracker"
                "<li> encounter builder"
                "<li> markdown-ish text editor"
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
        #     setScene on every update of combat stats. Also keep the fog mask
        #     and light range around. Remove once updates are piecemeal. 
        fogCenter = None
        fogMask = None
        lightRange = None
        gscene = self.gscene
        if (gscene is not None):
            if (gscene.getFogCenterLocked()):
                fogCenter = gscene.getFogCenter()
            fogMask = gscene.fog_mask
            lightRange = gscene.getLightRange() 

        # XXX Verify if all this old gscene cleaning up is necessary and/or
        #     enough
        if (gscene is not None):
            gscene.clear()
            gscene.setParent(None)
            gscene.fog_polys = []
        
        gscene = VTTGraphicsScene(scene)
        gscene.setFogColor(self.fogColor)
        if (fogCenter is not None):
            gscene.setFogCenterLocked(True)
            gscene.setFogCenter(fogCenter)
        if (fogMask is not None):
            gscene.fog_mask = fogMask
        if (lightRange is not None):
            gscene.setLightRange(lightRange)
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
        with (QSignalBlocker(gscene)):
            self.populateGraphicsScene(gscene, scene)
            self.graphicsView.setScene(gscene)
        
        self.player.stop()
        self.playlist.clear()
        for track in scene.music:
            logger.info("Adding music track %r", track.filepath)
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(track.filepath)))

        # XXX This is shared with the server thread, should be a deep copy and
        #     an atomic assignment
        global g_handouts
        g_handouts = scene.handouts

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

    def profile(self):
        import cProfile
        if (self.profiling):
            self.pr.disable()
            self.pr.dump_stats(os.path.join("_out", "profile.prof"))
            self.showMessage("Ended profiling")

        else:
            self.showMessage("Profiling...", 0)
            self.pr = cProfile.Profile()
            self.pr.enable()
            
        self.profiling = not self.profiling

    def newScene(self):
        scene = Struct()
        
        # XXX Create from an empty json so there's a single scene init path?
        #     Also, the "empty" json could have stock stuff like tokens, etc
        scene.map_doors = []
        scene.map_walls = []
        scene.cell_diameter = 70
        scene.cell_offset = [0, 0]
        scene.map_tokens = []
        scene.map_images = []
        scene.music = []
        scene.handouts = []
        scene.texts = []

        self.clearText()

        self.setScene(scene)

    def importDungeonScrawl(self):
        # XXX This overwrites the current walls and image, should warn if
        #     there's more than one image or the image filepath mismatches the
        #     final filepath?

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
        prefix = os_path_name(ds_filepath)
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
            # Don't try to copy to itself, it would create a zero sized file
            if (os.path.normpath(os_path_abspath(map_filepath)) != os.path.normpath(os_path_abspath(copy_filepath))):
                os_copy(map_filepath, copy_filepath)
                map_filepath = copy_filepath

        # Note the map filepath can be none if they decided to load only walls 
        import_ds(self.scene, ds_filepath, map_filepath)

        # Clear the fog of war after importing walls
        self.gscene.fog_mask = None

        self.setScene(self.scene, self.campaign_filepath)
    
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
            "name": os_path_name(filepath),
            # XXX Fix all the path mess for embedded assets
            "filepath": os.path.relpath(filepath),
            "hidden" : False,
            "ruleset_info" : Struct(**default_ruleset_info)
        })
        self.scene.map_tokens.append(s)
        # XXX Use something less heavy handed than setScene
        self.setScene(self.scene, self.campaign_filepath)

    def importImage(self, checked=False, filepath = None):
        # XXX This is called explicitly and as a shortcut handler/action, the
        #     second passes checked as second parameter so any extra optional
        #     parameters need to be passed as third parameter or later, split in
        #     two?
        if (filepath is None):
            dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
            # XXX Get supported extensions from Qt
            filepath, _ = QFileDialog.getOpenFileName(self, "Import Image", dirpath, "Images (*.png *.jpg *.jpeg *.jfif *.webp)")

            if (filepath == ""):
                filepath = None

            if (filepath is None):
                return

        # Try to guess the number of cells \xD7 is "multiplicative x"
        m = re.match(r".*\D+(\d+)\W*[x|\xD7]\W*(\d+)\D+", filepath)
        if (m is not None):
            img_size_in_cells = (int(m.group(1)), int(m.group(2)))
            logger.debug("img size in cells %s", img_size_in_cells)

        else:
            def mcd(a, b):
                res = 1
                for i in xrange(1, min(a, b) + 1):
                    if (((a % i) == 0) and ((b % i) == 0)):
                        res = i

                return res

            size = QImage(filepath).size()
            cellDiameterInPixels = mcd(size.width(), size.height())
            text, ok = QInputDialog.getText(
                self,
                "Image size in cells (%dx%d)" % (size.width(), size.height()), 
                "Cells (width, height):", QLineEdit.Normal, 
                "%d, %d" % (size.width()/cellDiameterInPixels, size.height()/cellDiameterInPixels)
            )
            if ((not ok) or (text == "")):
                return
            img_size_in_cells = [float(i) for i in text.split(",")]
            # If only one cell dimension was entered, guess the other one
            if (len(img_size_in_cells) == 1):
                cellDiameterInPixels = size.width() / img_size_in_cells[0]
                img_size_in_cells = [img_size_in_cells[0], size.height() / cellDiameterInPixels]


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

    def importHandout(self):
        # XXX Make it easier to share VTT internal media (from documentation
        #    browser, map screenshot, etc?)
            
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        # XXX Get supported extensions from Qt
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Handout", dirpath, "Handouts (*.png *.jpg *.jpeg *.jfif *.webp *.pdf *.doc *.docx)")

        if ((filepath == "") or (filepath is None)):
            return

        s = Struct(**{
            # XXX Fix all the path mess for embedded assets
            "filepath" :  os.path.relpath(filepath), 
            "name" : os.path.basename(filepath),
            "shared" : True,
        })
        self.scene.handouts.append(s)

        # XXX Use something less heavy handed
        self.setScene(self.scene, self.campaign_filepath)

    def importText(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        # XXX Get supported extensions from Qt
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Text", dirpath, "Handouts (*.txt *.rtf *.md *.html *.htm)")

        if ((filepath == "") or (filepath is None)):
            return

        s = Struct(**{
            # XXX Fix all the path mess for embedded assets
            "filepath" :  os.path.relpath(filepath), 
            "name" : os.path.basename(filepath),
        })

        
        self.scene.texts.append(s)

        self.openText(s.filepath)

        # Update the tree and anything that depends on scene.texts
        # XXX Use something less heavy handed
        self.setScene(self.scene, self.campaign_filepath)

    def importMusic(self):
        dirpath = os.path.curdir if self.campaign_filepath is None else os.path.dirname(self.campaign_filepath)
        # XXX Get supported extensions from Qt
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Music Track", dirpath, "Music Files (*.m4a *.mp3 *.ogg)")

        if (filepath == ""):
            filepath = None

        if (filepath is None):
            return

        # XXX Allow music area, etc
        # XXX Get metadata from music
        s = Struct(**{
            # XXX Fix all the path mess for embedded assets
            "filepath" :  os.path.relpath(filepath), 
            "name" : os_path_name(filepath),
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
            # XXX This only supports tokens
            js = json.dumps({ "tokens" :  map_tokens, "cell_diameter" : self.scene.cell_diameter }, indent=2, cls=JSONStructEncoder)
            logger.info("Copied to clipboard %s", js)
            qApp.clipboard().setText(js)

        else:
            # Copy as HTML
            # XXX Probably move to some table "export as HTML" menu?
            widget = self.focusWidget()
            if (isinstance(widget, VTTTableWidget)):
                table = widget
                if (isinstance(table.parent(), CombatTracker)):
                    tracker = table.parent()
                    # Note Trilium needs <html><body>  </body></html> bracketing
                    # or it won't recognize the html format (pasting onto Qt
                    # textedit doesn't need it, but it could be using OLE data?)
                    htmlLines = ["<html><body><table><thead style=\"background-color: black; color: white; weight:bold;\"><tr>"]

                    headers = [
                        # XXX Missing "XP", 
                        "Name", "HD", "HP", "HP_", "AC", "MR", "T0", "#AT", 
                        "Damage", "Alignment", "Notes"
                    ]
                    for header in headers:
                        # Note QTextEdit uses white on white as header color, so
                        # these won't be visible there
                        htmlLines.append('<th>%s</th>' % header)
                    htmlLines.append("</tr></thead><tbody>")

                    # Copy all table if there's no selection or only a single
                    # cell is selected
                    ranges = table.selectedRanges()
                    copyAllRows = (
                        len(ranges) == 0) or (
                        (len(ranges) == 1) and (ranges[0].rowCount() == 1) and 
                        (ranges[0].columnCount() == 1)
                    )
                    for j in xrange(table.rowCount()):
                        # Note can't check item.isSelected in the header loop
                        # below because the selected cell may not be one of the
                        # copied cells
                        copyThisRow = copyAllRows | any([
                            table.item(j,i).isSelected() for i in xrange(table.columnCount())])
                        rowLines = ["<tr>"]
                        for header in headers:
                            item = table.item(j, tracker.headerToColumn[header])
                            rowLines.append("<td>%s</td>" % item.text())
                        rowLines.append("</tr>")

                        if (copyThisRow):
                            htmlLines.extend(rowLines)
                            
                    htmlLines.append("</tbody></table></body></html>")
                    html = str.join("\n", htmlLines)
                    mimeData = QMimeData()
                    mimeData.setHtml(html)
                    qApp.clipboard().setMimeData(mimeData)

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
            gscene = self.gscene
            for item in gscene.selectedItems():
                logger.info("Deleting graphicsitem %s", item.data(0))
                if (gscene.isToken(item)):
                    self.scene.map_tokens.remove(item.data(0))

                elif (gscene.isGridHandle(item)):
                    pass
                
                elif (gscene.isWallHandle(item)):
                    wallHandle = item
                    wallItem = gscene.getWallItemFromWallHandleItem(wallHandle)
                    i = gscene.getPointIndexFromWallHandleItem(wallHandle)
                    map_wall = wallItem.data(0)
                    
                    # If ctrl is pressed remove just this point and join
                    # previous and next points, otherwise split into two walls
                    # at the deleted point

                    # XXX This is not a good place to check for modifiers, when
                    #     called from a shortcut unless the shortcut is duplicated
                    #     for the modifier and non modifier versions. Also won't
                    #     work when called from ctrl+x shortcut as part of cut
                    #     to clipboard
                    if (int(qApp.keyboardModifiers() & Qt.ControlModifier) != 0):
                        # Remove just this point
                        map_wall.points.pop(i)

                    else:
                        points = map_wall.points
                        self.scene.map_walls.remove(map_wall)
                        if (map_wall.closed):
                            # Turn into a single wall, open at the deleted point
                            startPoints = points[i+1:] + points[0:i] 
                            endPoints = []

                        else:
                            # Split into two open walls, split at the deleted
                            # point
                            startPoints = points[0:i]
                            endPoints = points[i+1:]
                        
                        # Ignore any resulting wall length 1 or less
                        if (len(startPoints) > 1):
                            self.scene.map_walls.append(Struct(points=startPoints, closed=False))

                        if (len(endPoints) > 1):
                            self.scene.map_walls.append(Struct(points=endPoints, closed=False))
            
                elif (gscene.isDoor(item)):
                    self.scene.map_doors.remove(item.data(0))

                else:
                    assert False, "Unknown graphics item %s" % item

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

    def clearFog(self):
        if (QMessageBox.question(self, "Clear fog of war", 
            "Are you sure you want to clear fog of war?") != QMessageBox.Yes):
            return
        self.gscene.fog_mask = None

        self.updateImage()
        
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
        dock.installEventFilter(self)

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
            if (int(qApp.keyboardModifiers() & Qt.ControlModifier) != 0):
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

        tracker.setScene(self.scene)
        
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

    def displayInitiativeOrder(self):
        logger.info("display %s", self.displayInitiativeOrderAct.isChecked())
        # Note the action has already been toggled at this point
        display = self.displayInitiativeOrderAct.isChecked()
        for tracker in self.findChildren(CombatTracker):
            tracker.setDisplayInitiativeOrder(display)
            
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
        logger.info("from %s to %s", class_name(old), class_name(new))
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
                /* Setting the title text color has no effect, set the global color instead */
                QDockWidget { color: palette(highlighted-text);
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
        """
        Focus the given dock (or the central widget if dock is the central widget)
        """
        logger.info("Focusing %r %s", dock.windowTitle(), class_name(dock))
        
        # XXX Missing focusing on tab click, see https://stackoverflow.com/questions/51215635/notification-when-qdockwidgets-tab-is-clicked
        #     It's not clear if the above can be used when there are multiple
        #     dockwidgets, the tabbar doesn't seem to have any pointers to the
        #     dock widget?
        # XXX Clicking on empty encounter table swallows ctrl+tab navigation, investigate
        
        # Don't do dock stuff if focusing the central widget
        if (isinstance(dock, QDockWidget)):
            dock.show()
            dock.raise_()
            widget = dock.widget()
            
        else:
            # Don't focus the centralWidget since it will focus on that widget
            # instead of a focusable child and it's not clear how to make it
            # focus on the proper child automatically (looks like focusing on
            # the central widget programmatically won't focus on the first
            # focusable child, an extra tab keypress is needed to focus it).
            # Trying to find the first focusable child with nextInFocusChain
            # doesn't seem to work
            
            # XXX Find out what the problem is, don't hardcode and assume the
            #     centralWidget is the docEditor
            widget = dock.textEdit
            
        widget.setFocus(Qt.TabFocusReason)

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
        logger.info("Focused widget %r %s", self.focusWidget().windowTitle(), class_name(self.focusWidget()))
        focusedDock = self.findFocusedDock()

        centralWidget = self.centralWidget()
        if (focusedDock is None):
            focusedDock = centralWidget
            
        logger.info("Focused dock %r", focusedDock.windowTitle())

        prev = None
        docks = [centralWidget] + self.getDocksInTabOrder()
        for dock in docks:
            logger.info("Found dock %r focus %s" % (dock.windowTitle(), dock.hasFocus()))
            if (dock is focusedDock):
                if (prev is None):
                    prev = docks[-1]
                self.focusDock(prev)
                
                break
            prev = dock

    def nextWindow(self):
        logger.info("Focused widget %r %s", self.focusWidget().windowTitle(), class_name(self.focusWidget()))
        focusedDock = self.findFocusedDock()

        centralWidget = self.centralWidget()
        if (focusedDock is None):
            focusedDock = centralWidget

        logger.info("Focused dock %r", focusedDock.windowTitle())

        prev = None
        docks = [centralWidget] + self.getDocksInTabOrder()
        for dock in docks:
            logger.info("Found dock %r focus %s" % (dock.windowTitle(), dock.hasFocus()))
            if ((prev is focusedDock) or (dock is docks[-1])):
                if (prev is not focusedDock):
                    dock = docks[0]
                self.focusDock(dock)
                    
                break
            prev = dock
        
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
        # XXX Remove legacy conversion once all maps have door objects
        if (len(js["map_doors"]) == 0):
            scene.map_doors = []

        elif (isinstance(js["map_doors"][0], list)):
            # XXX This needs to convert to the new polyline door
            scene.map_doors = [Struct(lines=door, open=False) for door in js["map_doors"]]
        
        elif ("lines" in js["map_doors"][0]):
            # Convert from independent lines of scalar coordinates, to polylines
            # of pairs of coordinates (add the first, last, and every other
            # point, each point is 2 coordinates)
            #
            # Leave implicitly closed (first point is not replicated at the end)
            
            # XXX Remove once all qvt files have been converted from "lines"
            #    (list of coordinates) for independent lines to "points" (list
            #    of polyline points)
            scene.map_doors = [Struct(
                points=[
                    [map_door["lines"][(i*4)], map_door["lines"][(i*4+1)]] for i in xrange(len(map_door["lines"])/4)
                ] + [[map_door["lines"][-2], map_door["lines"][-1]]],
                open=map_door["open"]) for map_door in js["map_doors"] 
            ]
            
        else:
            scene.map_doors = [Struct(**map_door) for map_door in js["map_doors"]]

        # Convert wall lines to wall objects
        scene.map_walls = js["map_walls"]
        if ((len(js["map_walls"]) > 0) and (isinstance(js["map_walls"][0], list))):
            # Legacy files have independent lines as lists of 4 coordinates
            # instead of structs
            scene.map_walls = [Struct(points=[walls[0:2], walls[2:4]], closed=False) for walls in js["map_walls"]]
        
        else:
            scene.map_walls = [Struct(**map_wall) for map_wall in js["map_walls"]]

        scene.cell_diameter = js["cell_diameter"]
        scene.cell_offset = js.get("cell_offset", [0, 0])
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
        scene.handouts = [Struct(**handout) for handout in js.get("handouts", [])]
        scene.texts = [Struct(**text) for text in js.get("texts", [])]

        self.setScene(scene, filepath)

        # Load the first text in the docEditor
        # XXX This only makes sense if the central widget is used as docEditor,
        #     otherwise it should be done with the dockwidget save & restore 
        if (len(scene.texts) > 0):
            self.openText(scene.texts[0].filepath)

        else:
            self.clearText()

        self.showMessage("Loaded %r" % filepath)

    def loadDysonUrl(self, url):
        """
        eg https://dysonlogos.blog/2021/01/08/dungeons-of-the-grand-illusionist/
        """
        import urllib2
        import contextlib
        
        with contextlib.closing(urllib2.urlopen(url)) as f:
            data = f.read()

        nest_count = 0
        entry_nest_count = None
        entry_start = None
        entry = None
        flair_start = None
        flair_nest_count = None
        flair_end = None
        image_nest_count = None
        images = []
        image_urls = []
        # XXX Use QXml, QDomDocument, or QXmlQuery instead of manual parsing?
        #     See http://3gfp.com/wp/2014/07/three-ways-to-parse-xml-in-qt/
        #     See http://3gfp.com/2015/01/qt-xml-parsing-continued/
        for m in re.finditer(r'<(?P<opentag>\w+)(?P<attribs>[^>]*)>|</(?P<closetag>\w+)>', data):
            opentag = m.group('opentag')
            if (opentag is not None):
                # Don't try to track every tag, some tags open and close in a
                # single angle or some tags don't close, which would disbalance
                # nest_count
                if (opentag in ["div", "h1"]):
                    nest_count += 1
                    if (opentag == "div"):
                        s = m.group('attribs')
                        if (s is not None):
                            # XXX Do better attrib value parsing, but keep in
                            #     mind needs proper quote grouping

                            # post-entry div contains the full article
                            if ("post-entry" in s):
                                entry_nest_count = nest_count
                                entry_start = m.start()

                            # jp-post-flair is a div inside post-entry div that
                            # has the social network links, etc, remove
                            elif ("jp-post-flair" in s):
                                flair_nest_count = nest_count
                                flair_start = m.start()
                            
                            # wp-block-image is the div that contains the link
                            # with the figure and img tags

                            # XXX Make this more flexible so it works with any
                            #     image tag/link inside post-entry? Eg see
                            #     https://dysonlogos.blog/2017/08/15/infested-hall-with-video/
                            elif ("wp-block-image" in s):
                                image_nest_count = nest_count
                                images.append([m.start(), None])
                    
                    elif (opentag == "h1"):
                        h1_start = m.start()

            else:
                closetag = m.group('closetag')
                if (closetag in ["div", "h1"]):
                    if (closetag == "div"):
                        if (nest_count == entry_nest_count):
                            # XXX All this assumes utf-8
                            encoding = 'utf-8'
                            entry = data[h1_start:h1_end].decode(encoding) + ('\n<a href="%s">%s</a>\n' % (url, url))
                            image_end = entry_start
                            
                            for image in images:
                                image_start = image[0]
                                entry += data[image_end:image_start].decode(encoding)
                                image_end = image[1]
                                img = data[image[0]:image[1]]
                                # XXX This could offer to choose the quality of
                                #     the embedded images (data-medium-file,
                                #     data-large-file)
                                mm = re.search(r'data-medium-file="([^"]*)"', img)
                                if (mm is not None):
                                    # Some random images don't have
                                    # data-medium-file, ignore
                                    image_url = mm.group(1)
                                    with contextlib.closing(urllib2.urlopen(image_url)) as f:
                                        image_data = f.read()

                                        data_url = bytes_to_data_url(image_data)
                                        entry += '<div align="center"><img src="%s"/></div>' % data_url

                                    # Add to list to let user choose later which
                                    # one to use as map
                                    mm = re.search(r'<a href="([^"]+)"', img)
                                    # Some images don't an have a link to the
                                    # high res version, skip
                                    if (mm is not None):
                                        image_url = mm.group(1)
                                        image_urls.append(image_url)

                                else:
                                    logger.warning("data-medium-file not found in %r", img)
                            
                            entry += data[image_end:flair_start].decode(encoding) + data[flair_end:m.end()].decode(encoding)
                            break
                        
                        elif (flair_nest_count == nest_count):
                            flair_end = m.end()
                            flair_nest_count = None

                        elif (image_nest_count == nest_count):
                            images[-1][1] = m.end()
                            image_nest_count = None
                    
                    elif (closetag == "h1"):
                        h1_end = m.end()
                
                    nest_count -= 1

        scene = Struct()
        
        # XXX Create from an empty json so there's a single scene init path?
        #     Also, the "empty" json could have stock stuff like tokens, etc
        scene.map_doors = []
        scene.map_walls = []
        scene.cell_diameter = 70
        scene.cell_offset = [0, 0]
        scene.map_tokens = []
        scene.map_images = []
        scene.music = []
        scene.handouts = []
        scene.texts = []

        self.setScene(scene)

        self.docEditor.textEdit.setHtml(entry)
        # Scroll to the end which normally has the dpi information so it can be
        # seen by the user when prompted for number of cells in the map
        self.docEditor.textEdit.moveCursor(QTextCursor.End)
        # Force to ask for name when saving
        self.docEditor.setFilepath(None)
        self.docEditor.setModified(True)
        # Update the modified indicator
        self.updateTextTitle(self.docEditor.textEdit, True)
        
        # Request user to choose an image of the ones in the page, default by
        # heuristics to second to last which is normally the highest quality
        # nogrid map
        if (len(image_urls) > 0):
            text, ok = QInputDialog.getItem(
                self,
                "Choose map image", 
                "Map Image URL", image_urls, len(image_urls) - 2, False
            )
            if ((not ok) or (text == "")):
                pass

            else:
                image_url = text
                filename = os.path.basename(image_url)
                filepath = os.path.join("_out", "maps", filename)
                with contextlib.closing(urllib2.urlopen(image_url)) as f_in, open(filepath, "wb") as f_out:
                    buffer = f_in.read()
                    f_out.write(buffer)
                self.importImage(None, filepath)

    def openDysonUrl(self):
        text, ok = QInputDialog.getText(
            self,
            "Open Dyson Map", 
            "URL:", QLineEdit.Normal, 
            "https://dysonlogos.blog/2021/01/08/dungeons-of-the-grand-illusionist/"
        )
        if ((not ok) or (text == "")):
            return

        self.loadDysonUrl(text)
            
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
        # XXX setScene at this point causes the focus to be lost on the DM view
        #     (and the player view to be reset), which is undesirable because eg
        #     gets lost on every stat modification (attack roll, hitpoint
        #     update, etc), as a quick workaround, there's a hack so locking the
        #     focus survives setscene
        if (False):
            # XXX No need to use these two because they are triggered by
            #     setScene below, put back once picemeal updates are in
            self.updateTree()
            self.updateImage()

        else:
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
        scene_item.setExpanded(True)

        scene_item.addChild(QTreeWidgetItem(["d: %d o: %s" % (self.scene.cell_diameter, self.scene.cell_offset)]))

        music = getattr(self.scene, "music", [])
        folder_item = QTreeWidgetItem(["Music (%d)" % len(music)])
        scene_item.addChild(folder_item)
        folder_item.setExpanded(True)
        for track in music:
            subfolder_item = QTreeWidgetItem(["%s" % track.name])
            subfolder_item.setData(0, Qt.UserRole, track)

            item = QTreeWidgetItem(["%s" % track.filepath])
            subfolder_item.addChild(item)
            
            folder_item.addChild(subfolder_item)
            
        handouts = getattr(self.scene, "handouts", [])
        folder_item = QTreeWidgetItem(["Handouts (%d)" % len(handouts)])
        scene_item.addChild(folder_item)
        folder_item.setExpanded(True)
        for handout in handouts:
            subfolder_item = QTreeWidgetItem(["%s%s" % ("*" if handout.shared else "", handout.name)])
            subfolder_item.setData(0, Qt.UserRole, handout)
            subfolder_item.setFlags(subfolder_item.flags() | Qt.ItemIsEditable)
            item = QTreeWidgetItem(["%s" % handout.filepath])
            subfolder_item.addChild(item)
            
            folder_item.addChild(subfolder_item)

        texts = getattr(self.scene, "texts", [])
        folder_item = QTreeWidgetItem(["Texts (%d)" % len(texts)])
        scene_item.addChild(folder_item)
        folder_item.setExpanded(True)
        for text in texts:
            # XXX This should star if the item has been modified?
            subfolder_item = QTreeWidgetItem([text.name])
            subfolder_item.setData(0, Qt.UserRole, text)
            subfolder_item.setFlags(subfolder_item.flags() | Qt.ItemIsEditable)
            item = QTreeWidgetItem(["%s" % text.filepath])
            subfolder_item.addChild(item)
            
            folder_item.addChild(subfolder_item)
        
        folder_item = QTreeWidgetItem(["Walls (%d)" % sum([len(wall.points) - 1 for wall in self.scene.map_walls])])
        scene_item.addChild(folder_item)
        for wall in self.scene.map_walls: 
            child = QTreeWidgetItem(["%s" % (wall.points,)])
            child.setData(0, Qt.UserRole, wall)

            folder_item.addChild(child)
            
        folder_item = QTreeWidgetItem(["Doors (%d)" % len(self.scene.map_doors)])
        scene_item.addChild(folder_item)
        for door in self.scene.map_doors: 
            child = QTreeWidgetItem(["%s%s" % ("*" if door.open else "", door.points)])
            child.setData(0, Qt.UserRole, door)

            folder_item.addChild(child)

        folder_item = QTreeWidgetItem(["Images (%d)" % len(self.scene.map_images)])
        scene_item.addChild(folder_item)
        folder_item.setExpanded(True)
        for image in self.scene.map_images:
            imageItem = self.gscene.imageAtData(image)
            subfolder_item = QTreeWidgetItem(["%s (%dx%d)" % (
                os.path.basename(image.filepath), 
                imageItem.pixmap().width(), imageItem.pixmap().height()
            )])
            subfolder_item.setData(0, Qt.UserRole, image)

            item = QTreeWidgetItem(["%s" % (image.scene_pos,)])
            subfolder_item.addChild(item)
            item = QTreeWidgetItem(["%s" % image.scale])
            subfolder_item.addChild(item)
            
            folder_item.addChild(subfolder_item)

        folder_item = QTreeWidgetItem(["Tokens (%d)" % len(self.scene.map_tokens)])
        scene_item.addChild(folder_item)
        folder_item.setExpanded(True)
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

        # Collect handouts as dicts instead of Structs
        filepaths |= set([handout.filepath for handout in d.get("handouts", [])])
        d["handouts"] = [vars(handout) for handout in d.get("handouts", [])]

        # XXX This assumes there's one and only one doceditor and that it has
        #     valid contents, save before saving the texts since if the text
        #     doesn't have a filepath, it will create a text entry
        self.saveText()

        # Collect text as dicts instead of Structs
        filepaths |= set([text.filepath for text in d.get("texts", [])])
        d["texts"] = [vars(text) for text in d.get("texts", [])]
        

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
        # Zip files are STORED by default (no compression), use DEFLATED
        # (compression)
        with zipfile.ZipFile(tmp_filepath, "w", zipfile.ZIP_DEFLATED) as f:
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
                    
                    ext = os_path_ext(filepath)
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
            scene_pos = qlist(self.graphicsView.mapToScene(QPoint(j, j) * scene.cell_diameter + qSizeToPoint(self.graphicsView.size()/2.0)))
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
        for i, map_wall in enumerate(scene.map_walls):
            gscene.addWall(map_wall)
            
    def populateDoors(self, gscene, scene):
        for map_door in scene.map_doors:
            gscene.addDoor(map_door)

    def populateGrid(self, gscene, scene):
        gscene.addGrid()

    def populateGraphicsScene(self, gscene, scene):

        # XXX This seems out of place but may need to be set before populating
        #     anything else, investigate moving to populateGrid?
        gscene.setCellOffsetAndDiameter(QPointF(*scene.cell_offset), scene.cell_diameter)
        
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
            # XXX This assumes a single background image
            sceneRect = gscene.imageAt(0).sceneBoundingRect()
            f.write('<image href="image.png" x="%f" y="%f" width="%f" height="%f"/>\n' %
                (sceneRect.x(), sceneRect.top(), sceneRect.width(), sceneRect.height()))

            for token_item in gscene.tokens():
                sceneRect = token_item.sceneBoundingRect()

                imformat = "PNG"
                pixItem = gscene.getTokenPixmapItem(token_item)
                pix = pixItem.pixmap()
                dataurl = qImageToDataUrl(pix, imformat)
                label_item = gscene.getTokenLabelItem(token_item)

                f.write('<g class="draggable-group" transform="translate(%f,%f) scale(%f)"><image href="%s" width="%f" height="%f"/><text fill="white" font-size="10" font-family="Arial, Helvetica, sans-serif" transform="translate(%f,%f) scale(%f)">%s</text></g>\n' %
                    (sceneRect.x(), sceneRect.top(), token_item.scale(), dataurl, pix.width(), pix.height(), 
                        label_item.pos().x(), label_item.pos().y(), label_item.scale(), label_item.toPlainText()))

                f.write('')

            for poly in gscene.fog_polys:
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
        
    def renderFogOfWar(self, qim, fog_scale, maskFormat):
        save_masks = False
        gscene = self.gscene
        sceneRect = gscene.sceneRect()
        # XXX Using sceneRect for image dimensions in pixels is bad if there's a
        #     lot of scaling done because of big cell sizes, this should look at
        #     the pixel sizes of the images and use that one (same for all the 
        #     places that try to derive a size in pixels from the scene)
        sceneSize = sceneRect.size().toSize()

        # XXX This assumes the fogcolor is black, it's used elsewhere
        #     to guarantee the map dimensions are not leaked, should have
        #     some code to change black to whatever fogColor
        assert qtuple(self.fogColor) == (0, 0, 0)
        
        
        # XXX Use something lighter to update the fog polys for fog of war,
        #     since this also generates the fog graphic items
        self.updateFog(False, False)

        if (g_disable_player_view):
            return
        
        # Perform the formula
        
        # back * (accmask - currentmask) * fade + (back + tokens) * currentmask
        # newaccmask = accmask + currentmask
        
        # Note the first formula can be changed to use newaccmask
        # instead of accmask which reduces the number of simultaneous
        # masks needed and also fixes a bug in Qt where doing a
        # difference of a .fill(Qt.black) fails to do anything

        # Note the back and back + tokens calculations need to be done
        # in one shot since any overdraw will get corrupted if there's
        # a composition mode other than overwrite (eg multiply or add)
        
        # XXX The fog mask needs to be saved with the scene? or as 
        #     alpha channel of the background image or store the tokens
        #     paths and recreate the fog mask at load time?
        # XXX Needs an option to clear the fog of war
        # XXX Needs tools to set/clear the fog of war manually
        #     (rectangles, circles, etc)
        fog_mask = gscene.fog_mask
        if ((fog_mask is None) or (fog_mask.size() != sceneSize * fog_scale)):
            logger.info("Creating fog_mask")
            fog_mask = QImage(sceneSize * fog_scale, maskFormat)
            gscene.fog_mask = fog_mask
            # There's a Qt bug for which operating on a fill(Qt.black)
            # fails to do anything, this is avoided by using fog_mask + 
            # current_mask instead of fog_mask below
            fog_mask.fill(Qt.black)

        acc_mask = fog_mask
        if (save_masks):
            acc_mask.save(os.path.join("_out", "acc_mask.png"))

        logger.info("Calculating current_mask")
        current_mask = QImage(acc_mask.size(), maskFormat)
        current_mask.fill(Qt.white)
        p = QPainter(current_mask)
        p.scale(fog_scale, fog_scale)
        p.translate(-sceneRect.topLeft())
        p.setBrush(QBrush(Qt.black))
        p.setPen(QPen(Qt.black))
        for poly in gscene.fog_polys:
            poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in poly])
            # XXX drawconvexpolygon takes 33%, drawImage 30%, and 10%
            #     render on fonda.qvt. Having render render fog polygons
            #     makes render take more time than drawconvexpolygon, so
            #     doesn't look like a solution?
            p.drawConvexPolygon(poly)
        p.end()

        if (save_masks):
            current_mask.save(os.path.join("_out", "current_mask.png"))

        logger.info("Calculating new_current_mask")
        p = QPainter(acc_mask)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.drawImage(0,0, current_mask)
        p.end()

        if (save_masks):
            acc_mask.save(os.path.join("_out", "new_acc_mask.png"))
        
        # difference = (accmask - currentmask)
        logger.info("Creating difference")
        difference = acc_mask.copy()
        logger.info("Calculating difference")
        p = QPainter(difference)
        p.setCompositionMode(QPainter.CompositionMode_Difference)
        p.drawImage(0, 0, current_mask)
        p.end()

        if (save_masks):
            difference.save(os.path.join("_out", "difference.png"))
        
        # scratch = back * fade * difference
        # Note back needs to be rendered in one shot without any
        # composition since the ordering of the items is not guaranteed
        # and would get the composition applied otherwise This means
        # that the difference calculation and multiply by back cannot be
        # merged into a single step and the difference needs to be
        # calculated first and then multiplied in a different step
        # If there was a single background with no grids, this could
        # merged with other steps with  
        #    item = gscene.imageAt(0) p.scale(item.scale(),
        #    item.scale()) p.drawPixmap(0, 0,
        #    gscene.imageAt(0).pixmap())
        # XXX Back * fade could be cached if only token change?

        logger.info("Creating scratch")
        scratch = QImage(difference.size(), maskFormat)
        logger.info("Calculating scratch")
        p = QPainter(scratch)
        gscene.setTokensVisible(False)
        gscene.render(p)
        p.scale(1.0, 1.0)
        # The rest of the scene state gets restored below
        gscene.setTokensVisible(True)
        p.setCompositionMode(QPainter.CompositionMode_Multiply)
        p.fillRect(scratch.rect(), Qt.gray)
        p.drawImage(0, 0, difference)
        p.end()

        if (save_masks):
            scratch.save(os.path.join("_out", "scratch.png"))

        # (tokens + back) * currentmask + scratch
        gscene.setTokensVisible(False, True)
        logger.info("Rendering %d scene items on %dx%d image", len(gscene.items()), qim.width(), qim.height())
        p = QPainter(qim)
        gscene.render(p)
        p.scale(1.0/fog_scale, 1.0/fog_scale)
        p.setCompositionMode(QPainter.CompositionMode_Multiply)
        p.drawImage(0, 0, current_mask)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.drawImage(0, 0, scratch)
        p.scale(1.0, 1.0)
        # This is necessary so the painter winds down before the pixmap
        # below, otherwise it crashes with "painter being destroyed
        # while in use"
        p.end()

        if (save_masks):
            qim.save(os.path.join("_out", "qim.png"))

    def updateImage(self):
        logger.info("")
        global g_img_bytes
        gscene = self.gscene
        fogCenter = gscene.getFogCenter()

        if (g_disable_player_view):
            return

        # XXX This allows downscaling the canvas, but ideally the original
        #     background image should come downscaled already and any fine
        #     features (grid, text) done at high resolution, since downscaling
        #     at this point is going to cause blurred text and grid
        img_scale = 1.0
        # XXX Scaling fog independently from the image allows downscaling mask
        #     operations while having readable token labels since tokens are not
        #     rendered on the fog of war. Unfortunately map features (eg
        #     furniture) are rendered on the fog of war and appear blurred. In
        #     addition, downscaling also causes sliver artifacts that don't go
        #     away because the fog of war is accumlated over time.
        fog_scale = 1.0 * img_scale
        gscene.setLockDirtyCount(True)

        # XXX May want to gscene.clearselection but note that it will need to be
        #     restored afterwards or the focus rectangle will be lost in the DM
        #     view when moving, switching to a different token, etc

        sceneRect = gscene.sceneRect()
        # Grow to 1x1 in case there's no scene
        sceneSize = sceneRect.size().toSize().expandedTo(QSize(1, 1))
        # RGB32 is significantly faster than RGB888, it's also recommended by
        # QImage.Format Qt documentation. UpdateImage takes 2.8x the time using
        # RGB888 than RGB32, specifically drawConvexPolygons takes 10x,
        # drawImage takes 3x, render 1.5x. The fillrate functions take slightly
        # more: fillRect 0.92x, save 0.85x
        imageFormat = QImage.Format_RGB32
        # There's no single 8-bit channel format in Qt 5.3 (Alpha8 and 8bit
        # grayscale are Qt 5.5), so also use RGB32 for the masks for the above
        # performance reasons
        maskFormat = imageFormat

        qim = QImage(sceneSize * img_scale, imageFormat)
        
        # If there's no fog center, there's no fog, just clear to background and
        # prevent leaking the map dimensions
        if (fogCenter is None):
            qim.fill(self.fogColor)

        else:
            use_fog_of_war = True
            use_svg = False
                
            if (use_svg):
                self.generateSVG()
                # Use the unfogged image for svg
                # XXX Implement svg fog, image fog or image fog + map
                pix = gscene.imageAt(0).pixmap()
                qim = QImage(pix)
            
            else:
                # Hide all DM user interface helpers
                # XXX Hiding seems to be slow, verify? Try changing all to
                #     transparent otherwise? zValue? Have a player and a DM
                #     scene?
                logger.info("hiding DM ui")
                gscene.setWallsVisible(False)
                gscene.setDoorsVisible(False)
                
                if ((len(gscene.fog_polys) == 0) or (sceneSize == QSize(1,1)) or (not use_fog_of_war)):
                    self.updateFog(True, False)
                    gscene.setTokensVisible(False, True)
                    logger.info("Rendering %d scene items on %dx%d image", len(gscene.items()), qim.width(), qim.height())
                    p = QPainter(qim)
                    gscene.render(p)
                    # This is necessary so the painter winds down before the pixmap
                    # below, otherwise it crashes with "painter being destroyed
                    # while in use"
                    p.end()

                elif (use_fog_of_war):
                    self.renderFogOfWar(qim, fog_scale, maskFormat)

                # Restore all DM user interface helpers
                logger.info("restoring DM ui")
                gscene.setWallsVisible(self.graphicsView.drawWalls)
                gscene.setDoorsVisible(True)
                gscene.setTokensVisible(True, True)

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
        g_img_bytes = ba.data()
            
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

    def updateTextTitle(self, editor, modified):
        """
        Remove or append the modified indicator on the title

        Note this works for the main editor but also for any editor in a dock
        window
        """
        dock = self.findParentDock(editor)
        if (dock is None):
            dock = self
        title = dock.windowTitle()
        if (not modified and title.endswith("*")):
            title = title[:-1]
            dock.setWindowTitle(title)
        
        elif (modified and not title.endswith("*")):
            title = title + "*"
            dock.setWindowTitle(title)

        # XXX Should this be set unconditionally to modified?
        if (not modified):
            self.docEditor.setModified(False)

    def clearText(self):
        logger.info("")
        self.docEditor.clearText()
        self.updateTextTitle(self.docEditor.textEdit, False)

    def openText(self, filepath):
        logger.info("%s", filepath)
        # XXX If docEditor is in a dock this could open in a new dock if ctrl
        #     is pressed like the DocBrowser, etc
        self.docEditor.loadText(filepath)
        # Loading the text in the editor sets the modified indicator, remove and
        # update title
        self.updateTextTitle(self.docEditor.textEdit, False)

    def saveText(self):
        logger.info("")

        if (not self.docEditor.modified()):
            logger.info("Not saving unmodified text %r", self.docEditor.getFilepath())
            return

        # If there's no filepath for this text and it has been modified, ask for
        # one, default to the campaign filename
        filepath = self.docEditor.getFilepath()
        if (filepath is None):
            filename = "campaign.html"
            if (self.campaign_filepath is not None):
                filename = os_path_name(self.campaign_filepath) + ".html"
            filepath = os.path.join("_out", "documents", filename)
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Text", filepath, "Text (*.html)")

            if (filepath is not None):
                # XXX This is called from saveScene, so there's no setScene
                #     and shouldn't be done here since other places (tree, etc)
                #     don't update
                s = Struct(**{
                    # XXX Fix all the path mess for embedded assets
                    "filepath" :  os.path.relpath(filepath), 
                    "name" : os.path.basename(filepath),
                })
                self.scene.texts.append(s)

        if ((filepath == "") or (filepath is None)):
            logger.info("Not saving None filepath")
            return

        self.docEditor.saveText(filepath)
        # Remove the modified indicator
        self.updateTextTitle(self.docEditor.textEdit, False)

    def treeItemActivated(self, item):
        logger.info("%s", item.text(0))

        data = item.data(0, Qt.UserRole)
        if (data in self.scene.texts):
            filepath = data.filepath
            self.openText(filepath)

        else:
            # If the tree text is an existing filename or the data has a filepath
            # field, open it with the associated app
            
            # XXX Initially this is a rule for handouts (name or filepath), but also
            #     works for other scene items that have a filepath field (tokens,
            #     images, tracks). Should check the type instead of blindly probing
            #     for filepaths?
            
            filepath = item.text(0)
            if (not os.path.exists(filepath)):
                filepath = getattr(item.data(0, Qt.UserRole), "filepath", None)
                if ((filepath is not None) and (not os.path.exists(filepath))):
                    filepath = None

            if (filepath is not None):
                logger.info("Launching %r with preferred app", filepath)
                self.showMessage("Launching %r" % filepath)
                qLaunchWithPreferredAp(filepath)

    def treeItemChanged(self, item, column):
        logger.info("%s %d", item.text(0), column)
        
        # XXX This assumes the only editable items are handouts
        # XXX Set other items as editable (eg tokens) and make this work for
        #     them

        # If the handout name starts with "*" then the handout is shareable,
        # otherwise it's not
        
        # XXX This should use something nicer like a "share this handout"
        #     context menu or editing the "shared" tree subitem
        handout = item.data(0, Qt.UserRole)
        handout.shared = item.text(0).startswith("*")

        if (handout.shared):
            handout.name = item.text(0)[1:]

        else:
            handout.name = item.text(0)
        
        # XXX Use something less heavy handed than setScene
        self.setScene(self.scene, self.campaign_filepath)

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
        logger.info("")
        gscene = self.gscene
        if (gscene.dirtyCount != self.sceneDirtyCount):
            self.sceneDirtyCount = gscene.dirtyCount
            # XXX These should go via signals
            # updateImage calls updateFog, always renders polys but only
            # recalculates fog if fog dirty count mismatches scene dirty count
            self.updateImage()
            # XXX This is too heavy-handed, since it resets the tree focus and
            #     fold state
            self.updateTree()
            self.updateCombatTrackers()
            # XXX Hook here the scene undo stack? (probably too noisy unless
            #     filtered by time or by diffing the scene?)

            fogCenter = gscene.getFogCenter()
            if (fogCenter is not None):
                fogCenter = fogCenter.x(), fogCenter.y()
                name = "????"
                scale = 0.0
                if (gscene.getFogCenterLocked()):
                    name = "LOCKED"
                    
                elif (gscene.isToken(gscene.focusItem())):
                    map_token = gscene.focusItem().data(0)
                    name = map_token.name
                    scale = map_token.scale
                    if (map_token.hidden):
                        name = "*" + name
                    
                s = "%s: %.01f,%.01f %.2f" % (name, fogCenter[0], fogCenter[1], scale)

            else:
                s = ""

            # XXX Assumes 5ft per cell
            s += " %s (%.2f)" % (
                gscene.getLightRangeName(gscene.getLightRange()), 
                gscene.getLightRange() * 5.0 / gscene.getCellDiameter()
            )
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
        logger.debug("source %r type %s", class_name(source), EventTypeString(event.type()))
        
        if ((event.type() == QEvent.KeyPress) and (source is self.imageWidget) and 
            (event.text() == "f")):
            # Fit to window on image widget
            logger.info("text %r key %d", event.text(), event.key())
            # XXX This should push the current zoom & pan and fit, then 
            #     restore zoom and pan when not fitting
            self.imageWidget.setFitToWindow(not self.imageWidget.fitToWindow)
            self.imageWidget.update()
        
        elif ((event.type() == QEvent.KeyPress) and isinstance(source, QDockWidget)):
            if (isinstance(source.widget(), (ImageWidget, VTTGraphicsView)) and
                (event.text() == "x")):
                # XXX This could maximize any dock, not just ImageWidget and
                #     VTTGraphicsView, but needs to check it doesn't interfere
                #     with text editing?

                # Toggle maximize / restore dock
                dock = source
                widget = dock.widget()
                if (dock.isFloating()):
                    dock.setFloating(False)
                    # Imagewidget loses focus when docking, bring it back
                    widget.setFocus(Qt.TabFocusReason)
                    if (isinstance(widget, ImageWidget)):
                        widget.setFitToWindow(True)

                else:
                    dock.setFloating(True)
                    dock.setWindowState(Qt.WindowMaximized)
                    if (isinstance(widget, ImageWidget)):
                        widget.setFitToWindow(True)

        elif ((event.type() == QEvent.MouseButtonPress) and 
            isinstance(source, QDockWidget) and (not source.widget().hasFocus())):
            # Focus on the dock if clicked on titlebar, note there's no api to
            # access the default titlebar, so just focus if clicked on the dock
            # but outside of the widget
            logger.info("Dock click %d,%d %r %s %s focused %s", event.x(), event.y(),
                source.windowTitle(), source.frameGeometry(), source.widget().frameGeometry(),
                source.widget().hasFocus())
            if (not source.widget().frameGeometry().contains(event.x(), event.y())):
                logger.info("Focusing dock widget %r", source.widget().windowTitle())
                self.focusDock(source)
            
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