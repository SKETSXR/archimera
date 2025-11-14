// server.js
// Node/Express backend that groups CAD + PDF uploads into a single "Asset" document.
// Requires: express, multer, mongoose, uuid
// npm install express multer mongoose uuid

const express = require('express');
const multer = require('multer');
const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');

const app = express();
const cors = require('cors');
app.use(cors());
const PORT = process.env.PORT || 4000;
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, 'uploads');

// Ensure upload dir exists
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// Connect to MongoDB
const MONGO_URI = process.env.MONGO_URI || 'mongodb://mongo_user:mongo_password@localhost:27019/my_database?authSource=admin';
mongoose.connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB error', err));

// ---------- Schemas ----------

const FileSchema = new mongoose.Schema({
  file_name: String,       // stored file name on disk
  original_name: String,   // original filename uploaded
  role: String,            // 'cad' | 'sketch' | 'other'
  size: Number,
  server_path: String,     // full server path
  download_url: String,    // optional public URL
  uploaded_at: { type: Date, default: Date.now },
  uploaded_by: String
}, { _id: false });

const AssetSchema = new mongoose.Schema({
  // grouping metadata
  bundle_id: { type: String, index: true, sparse: true }, // optional client-provided grouping id
  project_name: { type: String, index: true },
  client_name: { type: String, index: true },
  category: String,
  version: String,
  basename: String,    // grouping key (derived)
  files: [FileSchema],
  created_at: { type: Date, default: Date.now },
  updated_at: { type: Date, default: Date.now }
});

// helpful index for searches
AssetSchema.index({ project_name: 1, client_name: 1, category: 1, version: 1, basename: 1 });

const Asset = mongoose.model('Asset', AssetSchema);

// ---------- Multer storage ----------

const storage = multer.diskStorage({
  destination: (req, file, cb) => { cb(null, UPLOAD_DIR); },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext);
    const safeBase = base.replace(/[^a-z0-9_\-\.]/ig, '_');
    const final = `${safeBase}_${Date.now()}_${uuidv4()}${ext}`;
    cb(null, final);
  }
});
const upload = multer({ storage, limits: { fileSize: 200 * 1024 * 1024 } }); // 200MB limit

// parse json bodies
app.use(express.json());
// serve uploads
app.use('/uploads', express.static(UPLOAD_DIR));

// Serve static files from the frontend folder
app.use(express.static(path.join(__dirname, '../frontend')));

// Default route to serve the frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// ---------- Helpers ----------

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function deriveBasename(originalName) {
  if (!originalName) return 'unknown';
  const ext = path.extname(originalName);
  let base = path.basename(originalName, ext);
  // remove common suffixes often used for sketches/finals/revisions
  base = base.replace(/_?sketch$/i, '');
  base = base.replace(/_?final$/i, '');
  base = base.replace(/_?rev[_-]?\d*$/i, '');
  base = base.replace(/\s+$/, '');
  return base;
}

function detectRole(originalName, fileRolesMap) {
  if (fileRolesMap && fileRolesMap[originalName]) {
    return (String(fileRolesMap[originalName]) || '').toLowerCase();
  }
  if (!originalName) return 'other';
  if (/\.(dwg|dxf)$/i.test(originalName)) return 'cad';
  if (/\.(pdf)$/i.test(originalName)) return 'sketch';
  return 'other';
}

// ---------- Routes ----------

// Default route for '/'
app.get('/', (req, res) => {
  res.send('Welcome to the SKETS CAD backend server!');
});

/**
 * POST /api/upload
 * form fields:
 *  - project_name
 *  - client_name
 *  - category
 *  - uploaded_by
 *  - version
 *  - bundle_id (optional) : if provided, all files will be attached to that bundle (Asset)
 *  - file_roles (optional JSON): map of originalName -> role (e.g. {"plan.dwg":"cad","sketch.pdf":"sketch"})
 * files:
 *  - files (multiple)
 */
app.post('/api/upload', upload.array('files'), async (req, res) => {
  try {
    const { project_name, client_name, category, uploaded_by, version, bundle_id } = req.body;
    const fileRolesJson = req.body.file_roles || '{}';
    let fileRoles = {};
    try { fileRoles = JSON.parse(fileRolesJson); } catch (e) { /* ignore parsing errors */ }

    if (!project_name || !client_name || !category || !version) {
      return res.status(400).json({ ok: false, error: 'Missing required metadata (project_name|client_name|category|version)' });
    }
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ ok: false, error: 'No files uploaded' });
    }

    // Map uploaded files -> normalized objects
    const uploadedFiles = req.files.map(f => {
      const original = f.originalname;
      const role = detectRole(original, fileRoles);
      return {
        file_name: f.filename,
        original_name: original,
        role,
        size: f.size,
        server_path: path.join(UPLOAD_DIR, f.filename),
        download_url: `${req.protocol}://${req.get('host')}/uploads/${encodeURIComponent(f.filename)}`,
        uploaded_at: new Date(),
        uploaded_by: uploaded_by || 'Unknown'
      };
    });

    // If bundle_id provided: upsert single Asset where bundle_id matches
    if (bundle_id) {
      const filter = { bundle_id, project_name, client_name, category, version };
      const update = {
        $setOnInsert: { bundle_id, project_name, client_name, category, version, created_at: new Date(), basename: bundle_id },
        $set: { updated_at: new Date() },
        $push: { files: { $each: uploadedFiles } }
      };
      const doc = await Asset.findOneAndUpdate(filter, update, { upsert: true, returnDocument: 'after' });
      return res.json({ ok: true, count: uploadedFiles.length, assetIds: [doc._id.toString()], bundle_id });
    }

    // No bundle_id: group by CAD basenames primarily. Strategy:
    // 1) for each CAD file: derive basename and upsert an Asset (attach CAD file)
    // 2) for each non-CAD (PDF) file: attempt to find an existing Asset by basename match; if none, create a new asset for that PDF
    const cadItems = uploadedFiles.filter(f => f.role === 'cad');
    const sketchItems = uploadedFiles.filter(f => f.role === 'sketch');
    const otherItems = uploadedFiles.filter(f => f.role !== 'cad' && f.role !== 'sketch');

    const createdOrUpdated = [];

    async function upsertAssetForBasename(basename, filesToAttach) {
      const filter = { project_name, client_name, category, version, basename };
      const update = {
        $setOnInsert: { project_name, client_name, category, version, basename, created_at: new Date() },
        $set: { updated_at: new Date() },
        $push: { files: { $each: filesToAttach } }
      };
      const doc = await Asset.findOneAndUpdate(filter, update, { upsert: true, returnDocument: 'after' });
      return doc;
    }

    // 1) upsert driven by CAD items
    for (const cad of cadItems) {
      const basename = deriveBasename(cad.original_name);
      const doc = await upsertAssetForBasename(basename, [cad]);
      createdOrUpdated.push({ basename, assetId: doc._id.toString(), attached: 'cad' });
    }

    // 2) attach sketches to an existing matching asset by basename, or create new asset if none found
    for (const sketch of sketchItems) {
      const sketchBase = deriveBasename(sketch.original_name);
      // try exact basename match first (case-insensitive)
      let found = await Asset.findOne({
        project_name, client_name, category, version,
        basename: { $regex: `^${escapeRegExp(sketchBase)}$`, $options: 'i' }
      });
      // fallback: look for assets whose basename is contained in sketchBase or vice versa
      if (!found) {
        found = await Asset.findOne({
          project_name, client_name, category, version,
          $or: [
            { basename: { $regex: escapeRegExp(sketchBase), $options: 'i' } },
            { basename: { $regex: escapeRegExp(sketchBase.split(/[_\-\s]/)[0] || sketchBase), $options: 'i' } }
          ]
        });
      }
      if (found) {
        found.files.push(sketch);
        found.updated_at = new Date();
        await found.save();
        createdOrUpdated.push({ basename: found.basename, assetId: found._id.toString(), attached: 'sketch' });
      } else {
        const doc = await upsertAssetForBasename(sketchBase, [sketch]);
        createdOrUpdated.push({ basename: sketchBase, assetId: doc._id.toString(), attached: 'sketch_new' });
      }
    }

    // 3) attach any other files similarly (create new assets if necessary)
    for (const other of otherItems) {
      const base = deriveBasename(other.original_name);
      const doc = await upsertAssetForBasename(base, [other]);
      createdOrUpdated.push({ basename: base, assetId: doc._id.toString(), attached: 'other' });
    }

    return res.json({ ok: true, count: uploadedFiles.length, details: createdOrUpdated });
  } catch (err) {
    console.error('Upload error', err);
    return res.status(500).json({ ok: false, error: String(err) });
  }
});

// GET /api/files returns assets (supports basic filters: q, project, client, category, date_from, date_to)
app.get('/api/files', async (req, res) => {
    try {
        const { q, project, client, category, date_from, date_to, uploaded_by } = req.query;
        const filter = {};
        if (project) filter.project_name = project;
        if (client) filter.client_name = client;
        if (category) filter.category = category;
        if (uploaded_by) filter['files.uploaded_by'] = uploaded_by;

        if (q) {
            const re = new RegExp(escapeRegExp(q), 'i');
            filter.$or = [
                { basename: re },
                { 'files.original_name': re },
                { 'files.file_name': re },
                { project_name: re },
                { client_name: re }
            ];
        }

        if (date_from || date_to) {
            const dtFilter = {};
            if (date_from) dtFilter.$gte = new Date(date_from);
            if (date_to) dtFilter.$lte = new Date(date_to);
            filter['files.uploaded_at'] = dtFilter;
        }

        // Return assets with the files array
        const docs = await Asset.find(filter).sort({ updated_at: -1 }).lean();
        const out = docs.map(d => ({
            _id: d._id,
            bundle_id: d.bundle_id || null,
            project_name: d.project_name,
            client_name: d.client_name,
            category: d.category,
            version: d.version,
            basename: d.basename,
            created_at: d.created_at,
            updated_at: d.updated_at,
            files: d.files,
            preview_url: (d.files.find(f => /\.pdf$/i.test(f.original_name)) || {}).download_url || null
        }));
        res.json(out);
    } catch (err) {
        console.error('Error in /api/files:', err); // Log the error
        res.status(500).json({ ok: false, error: String(err) });
    }
});

app.listen(PORT, () => {
  console.log(`SKETS CAD backend running on port ${PORT}`);
});
