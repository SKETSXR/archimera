# SKETS CAD File Manager - Prototype
Minimal skeleton for the PRD provided. Contains a Node.js + Express backend and a simple frontend (static HTML/JS).

## Features
- Upload CAD-related files (.dwg, .dxf, .pdf, etc.) using multipart/form-data
- Stores files on server filesystem (uploads/) and metadata in MongoDB
- Browse uploaded files via a simple list endpoint
- Example MongoDB schema and API routes

## Quick start (local)
1. Install Node.js (v18+ recommended) and MongoDB (or use MongoDB Atlas).
2. Copy `.env.example` to `.env` and fill values.
3. Install dependencies and start server:
   ```bash
   cd backend
   npm install
   npm run start
   ```
4. Open `frontend/index.html` in a browser (or serve it via static server).

## Notes
- This is a minimal prototype to get started. Add authentication, RBAC, validation, file-server/NAS mounting, Electron wrapper, and production hardening in later phases.
