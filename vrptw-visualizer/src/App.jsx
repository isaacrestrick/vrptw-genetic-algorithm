import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Zap, TrendingDown, BarChart3, Layers, Eye, EyeOff, Github } from 'lucide-react';
import './App.css';

export default function App() {
  const [selectedView, setSelectedView] = useState('overview');
  const [visibleAnimations, setVisibleAnimations] = useState(new Set());
  const [visibleAnalysis, setVisibleAnalysis] = useState(new Set());

  const toggleAnimation = (index) => {
    const newVisible = new Set(visibleAnimations);
    if (newVisible.has(index)) {
      newVisible.delete(index);
    } else {
      newVisible.add(index);
    }
    setVisibleAnimations(newVisible);
  };

  const toggleAnalysis = (index) => {
    const newVisible = new Set(visibleAnalysis);
    if (newVisible.has(index)) {
      newVisible.delete(index);
    } else {
      newVisible.add(index);
    }
    setVisibleAnalysis(newVisible);
  };

  const runs = [
    {
      name: 'Basic Time-lapse',
      file: 'vrptw_timelapse.gif',
      initial: 955723.59,
      final: 345179.37,
      improvement: 63.9,
      duration: '27 seconds',
      fps: 3,
      frames: 80,
      size: '2.8 MB'
    },
    {
      name: 'Enhanced Multi-Metric',
      file: 'vrptw_enhanced_timelapse.gif',
      initial: 925238.54,
      final: 503078.45,
      improvement: 45.6,
      duration: '20 seconds',
      fps: 3,
      frames: 60,
      size: '3.0 MB'
    }
  ];

  const staticImages = [
    {
      name: 'Solution Evolution',
      file: 'vrptw_solution_comparison.png',
      description: '5-panel progression from Gen 0 to Gen 149',
      size: '170 KB'
    },
    {
      name: 'Detailed Analysis',
      file: 'vrptw_detailed_analysis.png',
      description: 'Comprehensive 6-panel analytics report',
      size: '233 KB'
    }
  ];

  const stats = [
    { label: 'Total Improvement', value: '63.9%', icon: TrendingDown },
    { label: 'Generations', value: '150', icon: Layers },
    { label: 'Population Size', value: '100', icon: BarChart3 },
    { label: 'Convergence Point', value: 'Gen 75', icon: Zap }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-red-500 to-red-600 rounded-lg">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">VRPTW Genetic Algorithm</h1>
                <p className="text-slate-400">Time-Lapse Visualization Dashboard</p>
              </div>
            </div>
            <a href="https://github.com" className="text-slate-400 hover:text-white transition">
              <Github className="w-6 h-6" />
            </a>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {stats.map((stat, idx) => {
            const Icon = stat.icon;
            return (
              <Card key={idx} className="bg-slate-800 border-slate-700 hover:border-slate-600 transition">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium text-slate-300">{stat.label}</CardTitle>
                    <Icon className="w-4 h-4 text-red-500" />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-white">{stat.value}</div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Tabs */}
        <Tabs defaultValue="animations" className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-slate-800 border border-slate-700">
            <TabsTrigger value="animations" className="data-[state=active]:bg-red-600">
              🎬 Animations
            </TabsTrigger>
            <TabsTrigger value="analysis" className="data-[state=active]:bg-red-600">
              📊 Analysis
            </TabsTrigger>
            <TabsTrigger value="info" className="data-[state=active]:bg-red-600">
              ℹ️ Information
            </TabsTrigger>
          </TabsList>

          {/* Animations Tab */}
          <TabsContent value="animations" className="space-y-6">
            <div className="space-y-4">
              {runs.map((run, idx) => (
                <Card key={idx} className="bg-slate-800 border-slate-700 overflow-hidden hover:border-red-500/50 transition">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-white flex items-center gap-2">
                          <span className="inline-block w-3 h-3 rounded-full bg-gradient-to-r from-red-500 to-red-600"></span>
                          {run.name}
                        </CardTitle>
                        <CardDescription className="text-slate-400 mt-1">
                          {run.duration} @ {run.fps} fps • {run.frames} frames • {run.size}
                        </CardDescription>
                      </div>
                      <Badge className="bg-red-600 text-white">{run.improvement}% Improvement</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-slate-700/50 rounded-lg p-3">
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Initial Fitness</p>
                        <p className="text-lg font-bold text-white mt-1">{run.initial.toLocaleString()}</p>
                      </div>
                      <div className="bg-slate-700/50 rounded-lg p-3">
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Final Fitness</p>
                        <p className="text-lg font-bold text-white mt-1">{run.final.toLocaleString()}</p>
                      </div>
                      <div className="bg-gradient-to-r from-red-600/20 to-red-500/20 rounded-lg p-3 border border-red-500/30">
                        <p className="text-xs text-slate-300 uppercase tracking-wide font-semibold">Total Gain</p>
                        <p className="text-lg font-bold text-red-400 mt-1">
                          -{((run.initial - run.final) / 1000).toFixed(0)}K
                        </p>
                      </div>
                    </div>
                    {visibleAnimations.has(idx) && (
                      <div className="mb-4 rounded-lg overflow-hidden border border-slate-600">
                        <img 
                          src={`http://localhost:8000/${run.file}`}
                          alt={run.name}
                          className="w-full h-auto"
                        />
                      </div>
                    )}
                    <div className="pt-2">
                      <Button 
                        className="w-full bg-red-600 hover:bg-red-700"
                        onClick={() => toggleAnimation(idx)}
                      >
                        {visibleAnimations.has(idx) ? (
                          <EyeOff className="w-4 h-4 mr-2" />
                        ) : (
                          <Eye className="w-4 h-4 mr-2" />
                        )}
                        {visibleAnimations.has(idx) ? 'Hide Animation' : 'View Animation'}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            <div className="space-y-4">
              {staticImages.map((img, idx) => (
                <Card key={idx} className="bg-slate-800 border-slate-700 overflow-hidden hover:border-red-500/50 transition">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-white">{img.name}</CardTitle>
                        <CardDescription className="text-slate-400 mt-1">{img.description}</CardDescription>
                      </div>
                      <Badge variant="outline" className="border-slate-600 text-slate-300">{img.size}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {visibleAnalysis.has(idx) && (
                      <div className="mb-4 rounded-lg overflow-hidden border border-slate-600">
                        <img 
                          src={`http://localhost:8000/${img.file}`}
                          alt={img.name}
                          className="w-full h-auto"
                        />
                      </div>
                    )}
                    <Button 
                      className="w-full bg-red-600 hover:bg-red-700"
                      onClick={() => toggleAnalysis(idx)}
                    >
                      {visibleAnalysis.has(idx) ? (
                        <EyeOff className="w-4 h-4 mr-2" />
                      ) : (
                        <Eye className="w-4 h-4 mr-2" />
                      )}
                      {visibleAnalysis.has(idx) ? 'Hide Analysis' : 'View Analysis'}
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Info Tab */}
          <TabsContent value="info" className="space-y-6">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">About This Project</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-slate-300">
                <div>
                  <h3 className="font-semibold text-white mb-2">🚛 Vehicle Routing Problem with Time Windows</h3>
                  <p>A complex combinatorial optimization problem where multiple vehicles must visit customers within specified time windows while minimizing total distance traveled.</p>
                </div>
                <div>
                  <h3 className="font-semibold text-white mb-2">🧬 Genetic Algorithm Solution</h3>
                  <p>Uses evolutionary computation with selection, crossover, and mutation operators to discover increasingly better routing solutions over 150 generations.</p>
                </div>
                <div>
                  <h3 className="font-semibold text-white mb-2">📊 Key Metrics</h3>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Population Size: 100 individuals</li>
                    <li>Generations: 150 evolution cycles</li>
                    <li>Mutation Rate: 15%</li>
                    <li>Crossover Rate: 85%</li>
                    <li>Selection: Tournament (size=5)</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">How to Use</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-slate-300 text-sm">
                <div>
                  <p className="font-mono bg-slate-700 rounded p-2 text-red-400">
                    python3 timelapse.py
                  </p>
                  <p className="mt-1">Generate basic time-lapse animation</p>
                </div>
                <div>
                  <p className="font-mono bg-slate-700 rounded p-2 text-red-400">
                    python3 enhanced_timelapse.py
                  </p>
                  <p className="mt-1">Generate enhanced multi-metric animation</p>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-r from-red-600/20 to-red-500/20 border-red-500/30">
              <CardHeader>
                <CardTitle className="text-white">📈 Results Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-slate-200 text-sm">
                <p>✅ Achieved 63.9% fitness improvement in first run</p>
                <p>✅ Rapid convergence in early generations (0-40)</p>
                <p>✅ Population diversity decreased as solutions matured</p>
                <p>✅ Generated 4 high-quality visualizations</p>
                <p>✅ Reusable, customizable codebase</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <div className="border-t border-slate-700 bg-slate-900/50 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
          <p className="text-center text-slate-400 text-sm">
            🚀 VRPTW Genetic Algorithm Visualizer • Built with React, Tailwind, and shadcn/ui
          </p>
        </div>
      </div>
    </div>
  );
}
