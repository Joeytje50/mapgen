import json
import os
import errno
from PIL import Image, ImageFilter, ImageEnhance
import sys
import errno
import glob
import numpy as np
from collections import defaultdict
import resource
import multiprocessing as mp

#Constants
PADDING = 64
FULLMAP = False
INTERMEDIATE = True
px_per_square = 4

def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

def mkdir_p(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('failed to mkdir', path, e)

def getBounds(regionList):
    lowX, lowY, highX, highY = 9999, 9999, 0, 0
    planes = 0
    for region in regionList:
        if 'xLowerLeft' in region: #typeA
            lowX = min(lowX, region['xUpperLeft'])
            highX = max(highX, region['xUpperRight'])
            lowY = min(lowY, region['yLowerLeft'])
            highY = max(highY, region['yUpperLeft'])
            planes = max(planes, region['numberOfPlanes'])
        elif 'newX' in region:
            lowX = min(lowX, region['newX'])
            highX = max(highX, region['newX'])
            lowY = min(lowY, region['newY'])
            highY = max(highY, region['newY'])
            planes = max(planes, region['numberOfPlanes'])
        elif 'xLow' in region:
            lowX = min(lowX, region['xLow'])
            highX = max(highX, region['xHigh'])
            lowY = min(lowY, region['yLow'])
            highY = max(highY, region['yHigh'])
            planes = max(planes, region['numberOfPlanes'])
        else:
            raise ValueError(region)
    return lowX, highX, lowY, highY, planes

def pointInsideBox(position, plane, lowX, highX, lowY, highY, chunk_lowX, chunk_highX, chunk_lowY, chunk_highY, allPlanes):
    x = position['x']
    y = position['y']
    z = position['z']
    lowX = lowX * 64 + chunk_lowX * 8
    lowY = lowY * 64 + chunk_lowY * 8
    highX = highX * 64 + chunk_highX * 8 + 7
    highY = highY * 64 + chunk_highY * 8 + 7
    return ((plane == 0) or (plane == z)) and x >= lowX and x <= highX and y >= lowY and y <= highY

def getIconsInsideArea(icons, plane, lowX, highX, lowY, highY, chunk_lowX=0, chunk_highX=7, chunk_lowY=0, chunk_highY=7, dx=0, dy=0, dz=0, allPlanes=False):
    valid = []
    for icon in icons:
        if pointInsideBox(icon['position'], plane, lowX, highX, lowY, highY, chunk_lowX, chunk_highX, chunk_lowY, chunk_highY, allPlanes):
            pos = icon['position']
            icon = [pos['x'] + dx, pos['y'] + dy, icon['spriteId']]
            valid.append(icon)
    return valid

def allBlack(im):
    data = np.asarray(im.convert('RGBA'))
    return np.count_nonzero(data[:,:,:3]) == 0

def buildImage(queue, im, defn, icons, version, plane, overallWidth, overallHeight):
    lowX, highX, lowY, highY, planes = getBounds(defn['regionList'])
    validIcons = []
    for region in defn['regionList']:
        if 'xLowerLeft' in region:
            oldLowX = region['xLowerLeft']
            oldHighX = region['xLowerRight']
            oldLowY = region['yLowerLeft']
            oldHighY = region['yUpperLeft']
            newLowX = region['xUpperLeft']
            newHighX = region['xUpperRight']
            newLowY = region['yLowerRight']
            newHighY = region['yUpperRight']
            print(oldLowX == newLowX, oldLowY == newLowY, oldHighX == newHighX, oldHighY == newHighY)
            validIcons.extend(getIconsInsideArea(icons, region['plane'] + plane, oldLowX, oldHighX, oldLowY, oldHighY, allPlanes=plane==0))
            for x in range(oldLowX, oldHighX + 1):
                for y in range(oldLowY, oldHighY + 1):
                    filename = "versions/{}/tiles/base/{}_{}_{}.png".format(version, region['plane'] + plane, x, y)
                    if os.path.exists(filename):
                        square = Image.open(filename)
                        imX = (x-lowX+newLowX-oldLowX) * px_per_square * 64
                        imY = (highY-y) * px_per_square * 64
                        im.paste(square, box=(imX+256, imY+256))
        elif 'chunk_oldXLow' in region:
            filename = "versions/{}/tiles/base/{}_{}_{}.png".format(version, region['oldPlane'] + plane, region['oldX'], region['oldY'])
            dx = region['newX'] * 64 + region['chunk_newXLow'] * 8 - region['oldX'] * 64 - region['chunk_oldXLow'] * 8
            dy = region['newY'] * 64 + region['chunk_newYLow'] * 8 - region['oldY'] * 64 - region['chunk_oldYLow'] * 8
            dz = 0 - region['oldPlane']
            validIcons.extend(getIconsInsideArea(icons, region['oldPlane'] + plane, region['oldX'], region['oldX'], region['oldY'], region['oldY'], region['chunk_oldXLow'], region['chunk_oldXHigh'], region['chunk_oldYLow'], region['chunk_oldYHigh'], dx, dy, dz, allPlanes=plane==0))
            if os.path.exists(filename):
                square = Image.open(filename)
                cropped = square.crop((region['chunk_oldXLow'] * px_per_square * 8,
                    (8-region['chunk_oldYHigh'] - 1) * px_per_square * 8,
                    (region['chunk_oldXHigh'] + 1) * px_per_square * 8,
                    (8-region['chunk_oldYLow']) * px_per_square * 8))
                imX = (region['newX']-lowX) * px_per_square * 64 + region['chunk_newXLow'] * px_per_square * 8
                imY = (highY-region['newY']) * px_per_square * 64 + (7-region['chunk_newYHigh']) * px_per_square * 8
                im.paste(cropped, box=(imX+256, imY+256))
        elif 'chunk_xLow' in region:
            validIcons.extend(getIconsInsideArea(icons, region['plane'] + plane, region['xLow'], region['xHigh'], region['yLow'], region['yHigh'], region['chunk_xLow'], region['chunk_xHigh'], region['chunk_yLow'], region['chunk_yHigh'], allPlanes=plane==0))
            filename = "versions/{}/tiles/base/{}_{}_{}.png".format(version, region['plane'] + plane, region['xLow'], region['yLow'])
            if os.path.exists(filename):
                square = Image.open(filename)
                cropped = square.crop((region['chunk_xLow'] * px_per_square * 8,
                    (8-region['chunk_yHigh'] - 1) * px_per_square * 8,
                    (region['chunk_xHigh'] + 1) * px_per_square * 8,
                    (8-region['chunk_yLow']) * px_per_square * 8))
                imX = (region['xLow']-lowX) * px_per_square * 64 + region['chunk_xLow'] * px_per_square * 8
                imY = (highY-region['yLow']) * px_per_square * 64 + (7-region['chunk_yHigh']) * px_per_square * 8
                im.paste(cropped, box=(imX+256, imY+256))
        elif 'xLow' in region:
            validIcons.extend(getIconsInsideArea(icons, region['plane'] + plane, region['xLow'], region['xHigh'], region['yLow'], region['yHigh'], allPlanes=plane==0))
            for x in range(region['xLow'], region['xHigh'] + 1):
                for y in range(region['yLow'], region['yHigh'] + 1):
                    filename = "versions/{}/tiles/base/{}_{}_{}.png".format(version, region['plane'] + plane, x, y)
                    if os.path.exists(filename):
                        square = Image.open(filename)
                        imX = (x-lowX) * px_per_square * 64
                        imY = (highY-y) * px_per_square * 64
                        im.paste(square, box=(imX+256, imY+256))
        else:
            raise ValueError(region)
    if queue:
        # return values in queue
        queue.put(im)
        queue.put(validIcons)
    #return validIcons

def layerBelow(im):
    im = ImageEnhance.Color(im).enhance(0.5) # 50% grayscale
    im = ImageEnhance.Brightness(im).enhance(0.7) # 30% darkness
    # im = ImageEnhance.Contrast(im).enhance(0.8) # 80% contrast
    # Do not perform blur here, because blurring with alpha channels is messy.
    return im

def buildMapID(defn, version, icons, iconSprites, baseMaps):
    mapId = -1
    if 'mapId' in defn:
        mapId = defn['mapId']
    elif 'fileId' in defn:
        mapId = defn['fileId']
    lowX, highX, lowY, highY, planes = getBounds(defn['regionList'])
    bounds = [[lowX * 64 - PADDING, lowY * 64 - PADDING], [(highX + 1) * 64 + PADDING, (highY + 1) * 64 + PADDING]]
    # bounds = [[0, 0], [12800, 12800]]
    if 'position' in defn:
        center = [defn['position']['x'], defn['position']['y']]
    else:
        print('cent')
        center = [(lowX + highX + 1) * 32, (lowY + highY + 1) * 32]
    baseMaps.put({'mapId': mapId, 'name': defn['name'], 'bounds': bounds, 'center': center})
    overallHeight = (highY - lowY + 1) * px_per_square * 64
    overallWidth = (highX - lowX + 1) * px_per_square * 64
    layersBelow = None
    for plane in range(planes):
        print(mapId, plane)
        im = Image.new("RGB", (overallWidth + 512, overallHeight + 512))

        # run this in a separate process to allow better cleaning up after running this function.
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(
            target=buildImage,
            args=(q, im, defn, icons, version, plane, overallWidth, overallHeight)
        )
        p.start()
        im = q.get()
        validIcons = q.get()
        p.join()

        if plane == 0:
            data = np.asarray(im.convert('RGB')).copy()
            data[(data == (255, 0, 255)).all(axis = -1)] = (0, 0, 0)
            im = Image.fromarray(data, mode='RGB')
            if planes > 1:
                layersBelow = layerBelow(im)
        elif plane > 0:
            data = np.asarray(im.convert('RGBA')).copy()
            data[:,:,3] = 255*(data[:,:,:3] != (255, 0, 255)).all(axis = -1)
            mask = Image.fromarray(data, mode='RGBA')
            im = layersBelow.convert("RGBA")
            # All filters except for blur are applied in layerBelow();
            # Apply blur just before pasting the current layer on top.
            im = im.filter(ImageFilter.GaussianBlur(radius=1))
            im.paste(mask, (0, 0), mask)
            if planes > plane and INTERMEDIATE:
                below = layerBelow(mask)
                layersBelow.paste(below, (0, 0), below)
        mem()

        for zoom in range(-3, 4):
            scalingFactor = 2.0**zoom/2.0**2
            zoomedWidth = int(round(scalingFactor * im.width))
            zoomedHeight = int(round(scalingFactor * im.height))
            zoomed = im.resize((zoomedWidth, zoomedHeight), resample=Image.BILINEAR)
            if zoom >= 0:
                for x, y, spriteId in validIcons:
                    sprite = iconSprites[spriteId]
                    width, height = sprite.size
                    imX = int(round((x - lowX * 64) * px_per_square * scalingFactor)) - width // 2 - 2
                    imY = int(round(((highY + 1) * 64 - y) * px_per_square * scalingFactor)) - height // 2 - 2
                    zoomed.paste(sprite, (imX+int(round(256*scalingFactor)), int(round(imY+256 * scalingFactor))), sprite)

            lowZoomedX = int((lowX - 1) * scalingFactor + 0.01)
            highZoomedX = int((highX + 0.9 + 1) * scalingFactor + 0.01)
            lowZoomedY = int((lowY - 1) * scalingFactor + 0.01)
            highZoomedY = int((highY + 0.9 + 1) * scalingFactor + 0.01)
            for x in range(lowZoomedX, highZoomedX + 1):
                for y in range(lowZoomedY, highZoomedY + 1):
                    coordX = int((x - (lowX - 1) * scalingFactor) * 256)
                    coordY = int((y - (lowY - 1) * scalingFactor) * 256)
                    cropped = zoomed.crop((coordX, zoomed.size[1] - coordY - 256, coordX + 256, zoomed.size[1] - coordY))
                    if not allBlack(cropped):
                        outfilename = "versions/{}/tiles/rendered/{}/{}/{}_{}_{}.png".format(version, mapId, zoom, plane, x, y)
                        mkdir_p(outfilename)
                        cropped.save(outfilename)
            # outfilename = "versions/{}/tiles/rendered/{}/{}_{}_full.png".format(version, mapId, plane, zoom)
            # mkdir_p(outfilename)
            # zoomed.save(outfilename)

#### MAIN ####

def main():
    if len(sys.argv) > 1:
        # run using: python3 stitch.py versions/2020-08-12_a/
        # easy peasy tab compleesy.
        version = sys.argv[1].lstrip('versions/').rstrip('/')
    else:
        version = "2020-08-12_a"

    with open("versions/{}/worldMapDefinitions.json".format(version)) as f:
        defs = json.load(f)

    with open("versions/{}/minimapIcons.json".format(version)) as f:
        icons = json.load(f)

    if FULLMAP:
        overallXLow = 999
        overallXHigh = 0
        overallYLow = 999
        overallYHigh = 0
        for file in glob.glob("versions/{}/tiles/base/*.png".format(version)):
            filename = file.split("/")[-1]
            filename = filename.replace(".png", "")
            plane, x, y = map(int, filename.split("_"))
            overallYHigh = max(y, overallYHigh)
            overallYLow = min(y, overallYLow)
            overallXHigh = max(x, overallXHigh)
            overallXLow = min(x, overallXLow)

        defs.append({"name": "debug", "mapId": -1, "regionList": [{"xLowerLeft": overallXLow, "yUpperRight": overallYHigh, "yLowerRight": overallYLow, "yLowerLeft": overallYLow, "numberOfPlanes": 4, "xUpperLeft": overallXLow, "xUpperRight": overallXHigh, "yUpperLeft": overallYHigh, "plane": 0, "xLowerRight": overallXHigh}]})


    iconSprites = {}
    for file in glob.glob("versions/{}/icons/*.png".format(version)):
        spriteId = int(file.split("/")[-1][:-4])
        iconSprites[spriteId] = Image.open(file)

    baseMaps = []
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    for defn in defs:
        p = ctx.Process(
            target=buildMapID,
            args=(defn, version, icons, iconSprites, q)
        )
        p.start()
        baseMaps.append(q.get())
        p.join()

    with open("versions/{}/basemaps.json".format(version, 'w')) as f:
        json.dump(baseMaps, f)

if __name__ == "__main__":
    main()
