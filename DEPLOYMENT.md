# Deployment Guide

This guide explains how to deploy the Node Localization Simulation Platform to various hosting platforms.

## 🚀 Render Deployment (Recommended)

### Prerequisites
- GitHub account with your repository
- Render account (free tier available)

### Steps to Deploy on Render

1. **Sign up/Login to Render**
   - Go to [render.com](https://render.com)
   - Sign up with your GitHub account

2. **Create New Static Site**
   - Click "New +" → "Static Site"
   - Connect your GitHub repository: `codingplugin/NL-Simulation`

3. **Configure Build Settings**
   - **Name**: `nl-simulation` (or your preferred name)
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
   - **Node Version**: `18.0.0`

4. **Environment Variables** (Optional)
   - No environment variables needed for this project

5. **Deploy**
   - Click "Create Static Site"
   - Render will automatically build and deploy your app
   - Your app will be available at: `https://your-app-name.onrender.com`

### Automatic Deployments
- Render automatically redeploys when you push to the `master` branch
- You can also manually trigger deployments from the Render dashboard

## 🌐 Alternative Deployment Options

### Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Netlify
1. Connect your GitHub repository
2. Build command: `npm run build`
3. Publish directory: `dist`

### GitHub Pages
1. Add to package.json:
```json
{
  "homepage": "https://yourusername.github.io/NL-Simulation",
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d dist"
  }
}
```

2. Install gh-pages:
```bash
npm install --save-dev gh-pages
```

3. Deploy:
```bash
npm run deploy
```

## 🔧 Build Configuration

### Vite Configuration
The project uses Vite for building. Key configuration in `vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser'
  }
})
```

### Build Process
1. **Development**: `npm run dev` (localhost:5173)
2. **Production Build**: `npm run build` (creates `dist/` folder)
3. **Preview Build**: `npm run preview` (localhost:4173)

## 📊 Performance Optimization

### Build Optimizations
- Code splitting is handled by Vite
- Assets are automatically optimized
- CSS is minified and optimized

### Runtime Optimizations
- React components are optimized for production
- Charts are lazy-loaded when needed
- Algorithms are tree-shaken for smaller bundle

## 🔍 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check Node.js version (>=16.0.0)
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Check for syntax errors in code

2. **Routing Issues**
   - Ensure `_redirects` file is in `public/` folder
   - Check that all routes redirect to `index.html`

3. **Performance Issues**
   - Optimize algorithm parameters
   - Reduce population size for large simulations
   - Use Web Workers for heavy computations

### Debug Commands
```bash
# Check build locally
npm run build
npm run preview

# Check bundle size
npm run build -- --analyze

# Test production build
npx serve dist
```

## 📈 Monitoring

### Render Analytics
- Render provides basic analytics
- Monitor build times and deployment status
- Check error logs in Render dashboard

### Performance Monitoring
- Use browser DevTools for performance analysis
- Monitor memory usage during large simulations
- Track algorithm convergence times

## 🔄 Continuous Deployment

### GitHub Actions (Optional)
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Render
on:
  push:
    branches: [master]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Render
        env:
          RENDER_TOKEN: ${{ secrets.RENDER_TOKEN }}
        run: |
          curl -X POST "https://api.render.com/v1/services/$SERVICE_ID/deploys" \
            -H "Authorization: Bearer $RENDER_TOKEN" \
            -H "Content-Type: application/json"
```

## 📞 Support

For deployment issues:
1. Check Render documentation: [docs.render.com](https://docs.render.com)
2. Review build logs in Render dashboard
3. Test locally with `npm run build && npm run preview`

## 🎯 Next Steps

After successful deployment:
1. Set up custom domain (optional)
2. Configure analytics (Google Analytics, etc.)
3. Set up monitoring and alerts
4. Optimize for mobile performance 