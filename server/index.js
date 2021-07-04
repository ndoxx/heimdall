const { execSync } = require("child_process");
const path = require('path');
const pointInPolygon = require('point-in-polygon');

const testOut = `
Done! Loaded 162 layers from weights-file
 Detection layer: 139 - type = 28
 Detection layer: 150 - type = 28
 Detection layer: 161 - type = 28
/home/ndx/dev/heimdall/proto/data/010.jpg: Predicted in 480.930000 milli-seconds.
car: 65%	(left_x:   26   top_y: 2954   width: 1363   height:  167)
car: 97%	(left_x:  266   top_y: 1730   width:  788   height:  453)
truck: 26%	(left_x: 1351   top_y: 1749   width:  590   height:  501)
car: 94%	(left_x: 1355   top_y: 1748   width:  573   height:  503)
car: 99%	(left_x: 2050   top_y: 1766   width:  459   height:  460)
car: 51%	(left_x: 2204   top_y: 1215   width:   88   height:   86)
`;

function detect(imagePath) {
    // Execute detection command and save output
    const cfgPath = 'cfg/yolov4.cfg';
    const wgtPath = 'data/yolov4.weights';
    const cmd = `build/darknet detect ${cfgPath} ${wgtPath} -ext_output ${imagePath}`;
    const output = execSync(cmd, 
        {
            cwd: '/home/ndx/git/darknet',
            stdio: 'pipe'
        }).toString('utf8');
    // const output = testOut;

    // Match all detection entry and extract results
    // Lines of interest are of the form: "car: 75%	(left_x:  239   top_y: 2310   width: 2036   height:  812)"
    const it = output.matchAll(/^(.+):\s*(\d+)%\s+\(left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)\)/gm);
    const vehicles = ['car', 'truck'];
    let objects = [];
    for (const match of it) {
        const objClass = match[1];
        if (!vehicles.includes(objClass))
            continue;

        const box = { leftX: parseInt(match[3]), topY: parseInt(match[4]), width: parseInt(match[5]), height: parseInt(match[6]) }
        objects.push({ objClass, score: parseInt(match[2]), box })
    }

    return objects;
}

function toCoordRepr(bb) {
    return { x1: bb.leftX, x2: bb.leftX + bb.width, y1: bb.topY - bb.height, y2: bb.topY };
}

const clamp = (num, min, max) => Math.min(Math.max(num, min), max);

// Compute the Jaccard index, or Intersection over Union of two AABBs
// This is a common metric for AABB overlapping
function IoU(box1, box2) {
    const bb1 = toCoordRepr(box1);
    const bb2 = toCoordRepr(box2);

    // Coordinates of intersection box
    const xLeft = Math.max(bb1.x1, bb2.x1);
    const yTop = Math.max(bb1.y1, bb2.y1);
    const xRight = Math.min(bb1.x2, bb2.x2);
    const yBottom = Math.min(bb1.y2, bb2.y2);

    if (xRight < xLeft || yBottom < yTop)
        return 0.0;


    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    const bb1Area = box1.width * box1.height;
    const bb2Area = box2.width * box2.height;
    const iou = intersectionArea / (bb1Area + bb2Area - intersectionArea);
    return clamp(iou, 0.0, 1.0);
}

function removeDoubleDetections(objects, threshold) {
    // For each pair of detected objects, if two of them have a well overlapping bounding box,
    // only keep the object with the biggest score
    let output = [];
    let overlaps = [];

    // Loop over non equal object pairs
    // If intersection over union of boxes greater than a threshold, mark for removal
    for (let ii = 0; ii < objects.length; ++ii) {
        for (let jj = ii + 1; jj < objects.length; ++jj) {
            if (IoU(objects[ii].box, objects[jj].box) > threshold)
                overlaps.push(objects[ii].score < objects[jj].score ? ii : jj);
        }
    }

    for (let ii = 0; ii < objects.length; ++ii) {
        if (overlaps.includes(ii))
            continue;
        output.push(objects[ii]);
    }

    return output;
}

function filterROI(objects, ROI) {
    let output = [];

    for (obj of objects) {
        // Compute centroid
        const cent = [obj.box.leftX + obj.box.width / 2, obj.box.topY + obj.box.height / 2];
        if (pointInPolygon(cent, ROI))
            output.push(obj);
    }

    return output;
}

const ROI_008 = [[3392, 1996], [3128, 1276], [1232, 1196], [672, 1712]];
const ROI_010 = [[2876, 2436], [2600, 1720], [220, 1684], [48, 2332]];

const pPath = path.join(__dirname, `../proto/data/010.jpg`);
let objects = detect(pPath)
objects = filterROI(objects, ROI_010);
objects = removeDoubleDetections(objects, 0.9);

console.log(objects);

const count = objects.length;

if (count < 4)
    console.log("GO");
else
    console.log("NO GO");