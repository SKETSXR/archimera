const mongoose = require('mongoose');
const fileSchema = new mongoose.Schema({
  project_name: String,
  category: String,
  file_name: String,
  uploaded_by: String,
  upload_date: { type: Date, default: Date.now },
  file_size: Number,
  server_path: String,
  version: String,
  original_name: String
}, { collection: 'files' });
module.exports = mongoose.model('File', fileSchema);
