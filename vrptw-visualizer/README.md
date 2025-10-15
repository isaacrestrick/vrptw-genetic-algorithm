# 🎬 VRPTW Genetic Algorithm Visualizer

A beautiful React dashboard showcasing time-lapse visualizations of the Vehicle Routing Problem with Time Windows (VRPTW) being solved by a Genetic Algorithm.

## ✨ Features

- **Modern Design**: Built with React, Tailwind CSS, and shadcn/ui components
- **Dark Theme**: Professional dark UI with gradient accents
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile
- **Multiple Visualizations**: 
  - 2 animated GIF time-lapses (2.8 MB + 3.0 MB)
  - 2 static analysis PNG images (170 KB + 233 KB)
- **Interactive Tabs**: Switch between animations, analysis, and information
- **Live Metrics**: Display key statistics and performance indicators
- **Beautiful Cards**: Clean component-based design using shadcn/ui

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation & Run

```bash
# Navigate to project directory
cd /Users/isaacrestrick/workspace/fractal/vrptw-visualizer

# Install dependencies (already done)
npm install

# Start development server
npm run dev
```

The app will be available at **http://localhost:5173** (or the URL shown in terminal)

## 📊 Dashboard Tabs

### 🎬 Animations Tab
View animated time-lapse visualizations:
- **Basic Time-lapse**: 27-second animation with dual panels
  - Left: Vehicle routes visualization
  - Right: Fitness evolution graph
  - 63.9% total improvement
  
- **Enhanced Multi-Metric**: 20-second animation with 5 panels
  - Solution routes
  - Fitness evolution
  - Per-generation improvements
  - Population diversity
  - Real-time statistics
  - 45.6% total improvement

### 📊 Analysis Tab
Static analysis images:
- **Solution Evolution**: 5-panel progression (Gen 0, 37, 75, 112, 149)
- **Detailed Analysis**: 6-panel comprehensive report with metrics table

### ℹ️ Information Tab
Educational content:
- About VRPTW problem
- How the genetic algorithm works
- Key metrics explanation
- Usage instructions
- Results summary

## 📁 Project Structure

```
vrptw-visualizer/
├── src/
│   ├── components/
│   │   └── ui/
│   │       ├── badge.jsx       # Badge component
│   │       ├── button.jsx      # Button component
│   │       ├── card.jsx        # Card component
│   │       └── tabs.jsx        # Tabs component
│   ├── lib/
│   │   └── utils.js            # Utility functions
│   ├── App.jsx                 # Main dashboard component
│   ├── App.css                 # App styles
│   └── index.css               # Global styles
├── package.json
├── tailwind.config.js          # Tailwind configuration
├── postcss.config.js           # PostCSS configuration
├── vite.config.js              # Vite configuration
└── index.html                  # HTML entry point
```

## 🎨 Design System

### Colors
- **Background**: Slate-900 with gradients
- **Accent**: Red-600 (primary CTA color)
- **Cards**: Slate-800 with slate-700 borders
- **Text**: White/Slate-300 for contrast

### Components Used
- **Tabs**: For section navigation
- **Card**: For data containers
- **Badge**: For highlighting metrics
- **Button**: For interactive actions

## 📈 Data Displayed

### Run 1 (Basic Time-lapse)
- Initial Fitness: 955,723.59
- Final Fitness: 345,179.37
- **Improvement: 63.9%**
- Convergence: Generation 75+

### Run 2 (Enhanced Multi-Metric)
- Initial Fitness: 925,238.54
- Final Fitness: 503,078.45
- **Improvement: 45.6%**
- Convergence: Generation 30+

### Algorithm Parameters
- Population Size: 100
- Generations: 150
- Mutation Rate: 15%
- Crossover Rate: 85%
- Selection: Tournament (size=5)

## 🔧 Available Commands

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint files (if configured)
npm run lint
```

## 🎯 Next Steps

### To View Visualizations
1. Click "View Animation" buttons in the Animations tab
2. Click "View Analysis" buttons in the Analysis tab
3. Images will open from the examples directory

### To Regenerate Visualizations
Navigate to the VRPTW examples directory:
```bash
cd /Users/isaacrestrick/workspace/fractal/vrptw-genetic-algorithm/examples

# Generate basic time-lapse
python3 timelapse.py

# Generate enhanced animation with metrics
python3 enhanced_timelapse.py
```

### To Customize
Edit `src/App.jsx` to:
- Add more visualization cards
- Change the styling and colors
- Add new tabs
- Integrate live image display

## 📦 Dependencies

### Core
- **react** (^18.x) - UI framework
- **vite** (^5.x) - Build tool

### UI Components & Styling
- **tailwindcss** - Utility-first CSS
- **shadcn/ui** - Accessible React components
- **tailwindcss-animate** - Animation utilities
- **class-variance-authority** - Component variants
- **clsx** - Utility for classnames
- **tailwind-merge** - Merge Tailwind classes
- **lucide-react** - Icon library

### Radix UI (Headless)
- **@radix-ui/react-tabs** - Tab component primitives

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Deploy Options
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy`
- **GitHub Pages**: Configure vite.config.js with `base`
- **Docker**: Create Dockerfile for containerization

### Environment Setup for Production
```bash
# Install production dependencies only
npm install --production

# Start production server
npm run preview
```

## 🎓 Learning Resources

### Understanding VRPTW
- See TIMELAPSE_README.md in examples directory
- See VISUALIZATION_SUMMARY.txt for detailed metrics

### React & Components
- Learn about React hooks and functional components
- Understand component composition with shadcn/ui

### Tailwind CSS
- Study the utility-first CSS approach
- Learn about responsive design (@media queries)

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9

# Or use different port
npm run dev -- --port 5174
```

### Build Issues
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Styling Issues
- Ensure tailwind.config.js has correct content paths
- Check that index.css imports @tailwind directives
- Verify classes are used correctly in JSX

## 📝 File Formats

The app references visualization files from:
```
/Users/isaacrestrick/workspace/fractal/vrptw-genetic-algorithm/examples/
```

Files referenced:
- `vrptw_timelapse.gif` (2.8 MB)
- `vrptw_enhanced_timelapse.gif` (3.0 MB)
- `vrptw_solution_comparison.png` (170 KB)
- `vrptw_detailed_analysis.png` (233 KB)

## 🤝 Contributing

To enhance the dashboard:

1. **Add new visualizations**:
   - Update `runs` or `staticImages` arrays in App.jsx
   - Add new tab sections for different content

2. **Create new components**:
   - Use shadcn/ui as a template
   - Place in `src/components/ui/`

3. **Customize styling**:
   - Edit Tailwind classes
   - Modify color scheme in `src/index.css`

## 📄 License

Same as parent project (MIT License)

---

## 🎉 Summary

This dashboard provides a beautiful, modern interface to showcase the VRPTW genetic algorithm's impressive improvements over 150 generations. The combination of animations and static analysis gives viewers a complete understanding of how evolutionary algorithms solve complex routing problems.

**Start the development server and explore the visualizations!** 🚀

---

**Built with ❤️ using React, Tailwind, and shadcn/ui**

*VRPTW Genetic Algorithm Visualizer • October 2025*
