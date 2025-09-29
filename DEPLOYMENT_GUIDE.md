# SIH Project Deployment Guide

## 1. Frontend (Vercel)
- Go to https://vercel.com/import
- Connect your GitHub repo (SIH)
- Vercel auto-detects React/Vite
- Click Deploy
- Copy your live site URL

## 2. Backend (Render) - Option A: Python Service
- Go to https://dashboard.render.com/new/web-service
- Connect your GitHub repo (SIH)
- Set root directory to `backend`
- Set start command: `gunicorn -b 0.0.0.0:$PORT app:app --workers 2 --timeout 120`
- Ensure `requirements.txt` is present in `backend/`
- Click Deploy
- Copy your backend URL

## 2. Backend (Render) - Option B: Docker Service  
- Go to https://dashboard.render.com/new/web-service
- Connect your GitHub repo (SIH)
- Leave root directory blank (uses repository root)
- Render will auto-detect the Dockerfile
- Click Deploy
- Copy your backend URL

## 3. Update Frontend API URLs
- In your frontend code, replace any `localhost` API URLs with your Render backend URL
- Example: `https://your-backend.onrender.com/api/kolam-from-image`

## 4. Test Your Live Site
- Visit your Vercel URL
- Upload images and verify Kolam generation

---

### Notes
- If you need to update API URLs, edit `frontend/src/pages/UploadPage.jsx` or wherever your API calls are made.
- For any issues, check Render and Vercel build logs for errors.
- Your backend must run from the `backend` directory and have all dependencies listed in `requirements.txt`.

---

**Ready to deploy!**
