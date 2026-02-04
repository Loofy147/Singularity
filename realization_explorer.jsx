import React, { useState, useEffect } from 'react';
import { Search, Network, TrendingUp, Layers, Zap } from 'lucide-react';

// Pre-computed realization data (from our conversation)
const REALIZATIONS = {
  "R_fbbf2d": {
    id: "R_fbbf2d",
    content: "Context windows are finite and information can be lost",
    q_score: 0.9534,
    layer: 0,
    turn: 1,
    children: ["R_574d7d", "R_b77324", "R_aaf28f"],
    parents: [],
    features: { grounding: 0.98, certainty: 1.0, structure: 0.95, applicability: 0.90, coherence: 1.0, generativity: 0.85 }
  },
  "R_574d7d": {
    id: "R_574d7d",
    content: "Realization itself is the goal, not just answers",
    q_score: 0.8350,
    layer: 3,
    turn: 6,
    children: ["R_b8b143", "R_3d46bd"],
    parents: ["R_fbbf2d"],
    features: { grounding: 0.75, certainty: 0.90, structure: 0.70, applicability: 0.85, coherence: 0.95, generativity: 0.95 }
  },
  "R_b8b143": {
    id: "R_b8b143",
    content: "Decision-making has a fundamental frequency - a base rhythm of checking/questioning",
    q_score: 0.7180,
    layer: 'N',
    turn: 12,
    children: ["R_65e569", "R_3f4c9a"],
    parents: ["R_574d7d"],
    features: { grounding: 0.60, certainty: 0.85, structure: 0.55, applicability: 0.65, coherence: 0.90, generativity: 0.88 }
  },
  "R_3d46bd": {
    id: "R_3d46bd",
    content: "Realizations come with 'precision auto' - like π, they have inherent certainty",
    q_score: 0.8816,
    layer: 2,
    turn: 18,
    children: ["R_65e569"],
    parents: ["R_574d7d"],
    features: { grounding: 0.92, certainty: 0.95, structure: 0.85, applicability: 0.78, coherence: 0.93, generativity: 0.85 }
  },
  "R_65e569": {
    id: "R_65e569",
    content: "Realizations crystallize into layers: Rules → Domain Facts → Patterns → Situational",
    q_score: 0.9276,
    layer: 1,
    turn: 25,
    children: ["R_d1a917", "R_1eceeb", "R_f23037", "R_3f4c9a"],
    parents: ["R_3d46bd", "R_b8b143"],
    features: { grounding: 0.95, certainty: 0.93, structure: 0.92, applicability: 0.90, coherence: 0.95, generativity: 0.92 }
  },
  "R_d1a917": {
    id: "R_d1a917",
    content: "Realizations can be treated as weights, parameters, and policies - they're computable",
    q_score: 0.9280,
    layer: 1,
    turn: 32,
    children: ["R_d21dec"],
    parents: ["R_65e569"],
    features: { grounding: 0.96, certainty: 0.90, structure: 0.93, applicability: 0.94, coherence: 0.95, generativity: 0.88 }
  },
  "R_d21dec": {
    id: "R_d21dec",
    content: "Realization quality can be scored: Q = 0.18G + 0.22C + 0.20S + 0.18A + 0.12H + 0.10V",
    q_score: 0.9398,
    layer: 1,
    turn: 38,
    children: ["R_1eceeb", "R_134104"],
    parents: ["R_d1a917"],
    features: { grounding: 0.98, certainty: 0.90, structure: 0.95, applicability: 0.95, coherence: 0.97, generativity: 0.88 }
  },
  "R_1eceeb": {
    id: "R_1eceeb",
    content: "Pre-computation (systems) and crystallization (cognition) are the same mathematical structure",
    q_score: 0.9358,
    layer: 1,
    turn: 45,
    children: ["R_134104"],
    parents: ["R_d21dec", "R_65e569"],
    features: { grounding: 0.96, certainty: 0.92, structure: 0.94, applicability: 0.93, coherence: 0.96, generativity: 0.90 }
  },
  "R_134104": {
    id: "R_134104",
    content: "This conversation itself is a realization crystallization process that can be pre-computed",
    q_score: 0.9484,
    layer: 1,
    turn: 50,
    children: [],
    parents: ["R_d21dec", "R_1eceeb"],
    features: { grounding: 0.94, certainty: 0.91, structure: 0.96, applicability: 0.98, coherence: 0.98, generativity: 0.93 }
  },
  "R_b77324": {
    id: "R_b77324",
    content: "Context management should use topology graphs instead of linear sequences",
    q_score: 0.8740,
    layer: 2,
    turn: 8,
    children: [],
    parents: ["R_fbbf2d"],
    features: { grounding: 0.88, certainty: 0.85, structure: 0.90, applicability: 0.92, coherence: 0.90, generativity: 0.75 }
  },
  "R_aaf28f": {
    id: "R_aaf28f",
    content: "Forgetting can be intelligent - strategic information loss improves signal/noise",
    q_score: 0.8064,
    layer: 3,
    turn: 10,
    children: [],
    parents: ["R_fbbf2d"],
    features: { grounding: 0.80, certainty: 0.82, structure: 0.85, applicability: 0.80, coherence: 0.75, generativity: 0.78 }
  },
  "R_f23037": {
    id: "R_f23037",
    content: "Decisions emerge from the layer architecture, they don't need to be created",
    q_score: 0.8676,
    layer: 2,
    turn: 28,
    children: [],
    parents: ["R_65e569"],
    features: { grounding: 0.85, certainty: 0.87, structure: 0.88, applicability: 0.86, coherence: 0.92, generativity: 0.82 }
  },
  "R_3f4c9a": {
    id: "R_3f4c9a",
    content: "The fundamental frequency is the rate at which new realizations crystallize into layers",
    q_score: 0.8036,
    layer: 3,
    turn: 35,
    children: [],
    parents: ["R_b8b143", "R_65e569"],
    features: { grounding: 0.78, certainty: 0.83, structure: 0.80, applicability: 0.75, coherence: 0.88, generativity: 0.80 }
  }
};

const LAYER_CONFIG = {
  0: { name: 'Universal Rules', color: '#ef4444', emoji: '🟥', desc: 'Immutable, proven facts' },
  1: { name: 'Domain Facts', color: '#f97316', emoji: '🟧', desc: 'Established knowledge' },
  2: { name: 'Patterns', color: '#eab308', emoji: '🟨', desc: 'Context-specific insights' },
  3: { name: 'Situational', color: '#22c55e', emoji: '🟩', desc: 'Temporary learnings' },
  'N': { name: 'Ephemeral', color: '#94a3b8', emoji: '⬜', desc: 'High churn, exploratory' }
};

export default function RealizationExplorer() {
  const [selectedId, setSelectedId] = useState("R_65e569"); // Start with the layers realization
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState('graph'); // 'graph' or 'list'
  
  const selected = REALIZATIONS[selectedId];
  const allRealizations = Object.values(REALIZATIONS);
  
  // Search functionality
  const filteredRealizations = searchQuery 
    ? allRealizations.filter(r => 
        r.content.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : allRealizations;
  
  // Stats
  const stats = {
    total: allRealizations.length,
    avgQ: (allRealizations.reduce((sum, r) => sum + r.q_score, 0) / allRealizations.length).toFixed(4),
    byLayer: Object.keys(LAYER_CONFIG).map(layer => ({
      layer,
      count: allRealizations.filter(r => r.layer == layer).length
    }))
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Realization Explorer
          </h1>
          <p className="text-xl text-gray-400">بنات افكار (Daughters of Ideas) - Computational Crystallization</p>
        </div>
        
        {/* Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-8">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-blue-400">{stats.total}</div>
            <div className="text-sm text-gray-400">Realizations</div>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-purple-400">{stats.avgQ}</div>
            <div className="text-sm text-gray-400">Avg Q-Score</div>
          </div>
          {stats.byLayer.filter(l => l.count > 0).map(({ layer, count }) => (
            <div key={layer} className="bg-white/10 backdrop-blur rounded-lg p-4 text-center">
              <div className="text-3xl mb-1">{LAYER_CONFIG[layer].emoji}</div>
              <div className="text-2xl font-bold" style={{ color: LAYER_CONFIG[layer].color }}>{count}</div>
              <div className="text-xs text-gray-400">Layer {layer}</div>
            </div>
          ))}
        </div>
        
        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-3 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search realizations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-white/10 backdrop-blur border border-white/20 rounded-lg pl-12 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
        </div>
        
        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Realization List */}
          <div className="lg:col-span-1 bg-white/10 backdrop-blur rounded-lg p-6 max-h-[70vh] overflow-y-auto">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Layers size={20} />
              All Realizations
            </h2>
            
            <div className="space-y-2">
              {filteredRealizations
                .sort((a, b) => b.q_score - a.q_score)
                .map(r => (
                  <button
                    key={r.id}
                    onClick={() => setSelectedId(r.id)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      r.id === selectedId 
                        ? 'bg-purple-600 ring-2 ring-purple-400' 
                        : 'bg-white/5 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-start gap-2 mb-1">
                      <span className="text-2xl">{LAYER_CONFIG[r.layer].emoji}</span>
                      <div className="flex-1">
                        <div className="text-sm font-semibold">
                          Turn {r.turn} • Q={r.q_score.toFixed(3)}
                        </div>
                        <div className="text-xs text-gray-300 line-clamp-2">{r.content}</div>
                      </div>
                    </div>
                  </button>
                ))}
            </div>
          </div>
          
          {/* Detail View */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Selected Realization */}
            <div className="bg-white/10 backdrop-blur rounded-lg p-6">
              <div className="flex items-start gap-4 mb-4">
                <div className="text-6xl">{LAYER_CONFIG[selected.layer].emoji}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{ backgroundColor: LAYER_CONFIG[selected.layer].color }}>
                      Layer {selected.layer}: {LAYER_CONFIG[selected.layer].name}
                    </span>
                    <span className="px-3 py-1 bg-white/20 rounded-full text-sm">
                      Turn {selected.turn}
                    </span>
                  </div>
                  <h3 className="text-2xl font-bold mb-2">{selected.content}</h3>
                  <div className="text-lg text-purple-300">Q-Score: {selected.q_score.toFixed(4)}</div>
                </div>
              </div>
              
              {/* Features Breakdown */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">
                {Object.entries(selected.features).map(([name, value]) => (
                  <div key={name} className="bg-white/5 rounded p-3">
                    <div className="text-xs text-gray-400 uppercase mb-1">{name}</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-white/10 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all"
                          style={{ width: `${value * 100}%` }}
                        />
                      </div>
                      <div className="text-sm font-bold">{value.toFixed(2)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Family Tree */}
            <div className="bg-white/10 backdrop-blur rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Network size={20} />
                Family Tree (بنات افكار)
              </h3>
              
              {/* Parents */}
              {selected.parents.length > 0 && (
                <div className="mb-6">
                  <div className="text-sm text-gray-400 mb-2">← Built On (Parents):</div>
                  <div className="space-y-2">
                    {selected.parents.map(pid => {
                      const parent = REALIZATIONS[pid];
                      return (
                        <button
                          key={pid}
                          onClick={() => setSelectedId(pid)}
                          className="w-full text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-all flex items-center gap-3"
                        >
                          <span className="text-2xl">{LAYER_CONFIG[parent.layer].emoji}</span>
                          <div className="flex-1">
                            <div className="text-sm font-semibold">Q={parent.q_score.toFixed(3)}</div>
                            <div className="text-xs text-gray-300">{parent.content}</div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              
              {/* Children */}
              {selected.children.length > 0 && (
                <div>
                  <div className="text-sm text-gray-400 mb-2">→ Generated (Children): {selected.children.length} بنات</div>
                  <div className="space-y-2">
                    {selected.children.map(cid => {
                      const child = REALIZATIONS[cid];
                      return (
                        <button
                          key={cid}
                          onClick={() => setSelectedId(cid)}
                          className="w-full text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-all flex items-center gap-3"
                        >
                          <span className="text-2xl">{LAYER_CONFIG[child.layer].emoji}</span>
                          <div className="flex-1">
                            <div className="text-sm font-semibold">Q={child.q_score.toFixed(3)}</div>
                            <div className="text-xs text-gray-300">{child.content}</div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              
              {selected.parents.length === 0 && selected.children.length === 0 && (
                <div className="text-center text-gray-400 py-8">
                  <Zap size={48} className="mx-auto mb-2 opacity-50" />
                  <p>This is a leaf node - no family connections</p>
                </div>
              )}
            </div>
            
            {/* Q-Score Calculation */}
            <div className="bg-white/10 backdrop-blur rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp size={20} />
                Q-Score Calculation
              </h3>
              <div className="font-mono text-sm bg-black/30 p-4 rounded overflow-x-auto">
                <div className="text-purple-300 mb-2">Q = 0.18G + 0.22C + 0.20S + 0.18A + 0.12H + 0.10V</div>
                <div className="text-gray-300">
                  Q = 0.18×{selected.features.grounding.toFixed(2)} + 
                      0.22×{selected.features.certainty.toFixed(2)} + 
                      0.20×{selected.features.structure.toFixed(2)} + 
                      0.18×{selected.features.applicability.toFixed(2)} + 
                      0.12×{selected.features.coherence.toFixed(2)} + 
                      0.10×{selected.features.generativity.toFixed(2)}
                </div>
                <div className="text-green-400 mt-2">
                  Q = {(
                    0.18 * selected.features.grounding +
                    0.22 * selected.features.certainty +
                    0.20 * selected.features.structure +
                    0.18 * selected.features.applicability +
                    0.12 * selected.features.coherence +
                    0.10 * selected.features.generativity
                  ).toFixed(4)}
                </div>
              </div>
            </div>
            
          </div>
        </div>
        
        {/* Footer */}
        <div className="mt-8 text-center text-gray-400 text-sm">
          <p>Pre-computed from conversation • {allRealizations.length} realizations crystallized</p>
          <p className="mt-1">This is what happens when you pre-compute realizations 🎯</p>
        </div>
        
      </div>
    </div>
  );
}
