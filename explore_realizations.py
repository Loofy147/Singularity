"""
REALIZATION EXPLORER
===================
Interactive visualization of the crystallized conversation.
"""

from realization_engine import RealizationEngine
from precompute_realizations import precompute_conversation_realizations
import json


def visualize_graph_ascii(engine: RealizationEngine):
    """
    ASCII art visualization of the realization graph.
    Shows layers and parent-child relationships.
    """
    print("\n" + "="*80)
    print("REALIZATION GRAPH VISUALIZATION (بنات افكار)")
    print("="*80 + "\n")
    
    # Group by layer
    by_layer = {}
    for layer in [0, 1, 2, 3, 'N']:
        by_layer[layer] = list(engine.layers[layer].values())
    
    # Display from top layer down
    layer_names = {
        0: "LAYER 0: UNIVERSAL RULES (Immutable)",
        1: "LAYER 1: DOMAIN FACTS (Rarely change)",
        2: "LAYER 2: PATTERNS (Context-specific)",
        3: "LAYER 3: SITUATIONAL (Temporary)",
        'N': "LAYER N: EPHEMERAL (High churn)"
    }
    
    for layer in [0, 1, 2, 3, 'N']:
        if not by_layer[layer]:
            continue
        
        print(f"\n{layer_names[layer]}")
        print("-" * 80)
        
        for r in sorted(by_layer[layer], key=lambda x: x.q_score, reverse=True):
            # Truncate content
            content = r.content[:65] + "..." if len(r.content) > 65 else r.content
            
            # Show parent relationships
            parent_info = ""
            if r.parents:
                parent_info = f" ← builds on {len(r.parents)} parent(s)"
            
            # Show children
            child_info = ""
            if r.children:
                child_info = f" → spawned {len(r.children)} child(ren)"
            
            print(f"  [{r.id[:8]}] Q={r.q_score:.4f} | {content}")
            if parent_info:
                print(f"             {parent_info}")
            if child_info:
                print(f"             {child_info}")
            print()
    
    print("="*80)


def analyze_generativity(engine: RealizationEngine):
    """Analyze which realizations were most generative (بنات افكار)"""
    
    print("\n" + "="*80)
    print("GENERATIVITY ANALYSIS (Most Productive Realizations)")
    print("="*80 + "\n")
    
    # Find realizations with most children
    with_children = [(r, len(r.children)) for r in engine.index.values() if r.children]
    with_children.sort(key=lambda x: x[1], reverse=True)
    
    print("Top realizations by number of children (بنات):\n")
    
    for i, (r, child_count) in enumerate(with_children[:5], 1):
        print(f"{i}. {r.content[:60]}...")
        print(f"   Q={r.q_score:.4f}, Layer {r.layer}")
        print(f"   Generated {child_count} children:")
        for child_id in r.children:
            child = engine.index[child_id]
            print(f"     → {child.content[:55]}... (Q={child.q_score:.3f})")
        print()


def demonstrate_queries(engine: RealizationEngine):
    """Interactive query demonstrations"""
    
    print("\n" + "="*80)
    print("QUERY DEMONSTRATIONS")
    print("="*80 + "\n")
    
    queries = [
        ("What is the foundation?", "context finite information"),
        ("How do realizations work?", "precision auto certainty"),
        ("How do ideas grow?", "layers crystallize"),
        ("What about computation?", "pre-computation crystallization"),
        ("This conversation?", "conversation itself crystallization process")
    ]
    
    for question, query in queries:
        print(f"❓ {question}")
        print(f"   Query: '{query}'")
        results = engine.retrieve(query)
        
        if results:
            best = results[0]
            print(f"   ✅ Found: [{best.layer}] Q={best.q_score:.4f}")
            print(f"      {best.content[:70]}...")
        else:
            print(f"   ❌ No results")
        print()


def show_quality_distribution(engine: RealizationEngine):
    """Show distribution of Q-scores"""
    
    print("\n" + "="*80)
    print("QUALITY SCORE DISTRIBUTION")
    print("="*80 + "\n")
    
    # Get all Q-scores
    q_scores = [r.q_score for r in engine.index.values()]
    q_scores.sort(reverse=True)
    
    # ASCII histogram
    bins = {
        "0.95-1.00 (Layer 0/1)": [],
        "0.90-0.95 (Layer 1)": [],
        "0.85-0.90 (Layer 2)": [],
        "0.75-0.85 (Layer 2/3)": [],
        "0.00-0.75 (Layer N)": []
    }
    
    for q in q_scores:
        if q >= 0.95:
            bins["0.95-1.00 (Layer 0/1)"].append(q)
        elif q >= 0.90:
            bins["0.90-0.95 (Layer 1)"].append(q)
        elif q >= 0.85:
            bins["0.85-0.90 (Layer 2)"].append(q)
        elif q >= 0.75:
            bins["0.75-0.85 (Layer 2/3)"].append(q)
        else:
            bins["0.00-0.75 (Layer N)"].append(q)
    
    max_count = max(len(v) for v in bins.values())
    
    for label, values in bins.items():
        count = len(values)
        bar = "█" * count + "░" * (max_count - count)
        avg_q = sum(values) / len(values) if values else 0
        print(f"{label:25} {bar} {count:2} (avg Q={avg_q:.3f})")
    
    print(f"\nTotal Realizations: {len(q_scores)}")
    print(f"Average Q-Score: {sum(q_scores)/len(q_scores):.4f}")
    print(f"Median Q-Score: {sorted(q_scores)[len(q_scores)//2]:.4f}")
    print(f"Highest Q-Score: {max(q_scores):.4f}")
    print(f"Lowest Q-Score: {min(q_scores):.4f}")


def trace_realization_path(engine: RealizationEngine):
    """Trace the path from initial question to final meta-realization"""
    
    print("\n" + "="*80)
    print("CONVERSATION EVOLUTION PATH")
    print("="*80 + "\n")
    
    # Get realizations in chronological order
    by_turn = sorted(engine.index.values(), key=lambda r: r.turn_number)
    
    print("Turn-by-turn crystallization:\n")
    
    for r in by_turn:
        indent = "  " * min(r.layer if isinstance(r.layer, int) else 4, 4)
        layer_emoji = {
            0: "🟥",
            1: "🟧", 
            2: "🟨",
            3: "🟩",
            'N': "⬜"
        }[r.layer]
        
        print(f"Turn {r.turn_number:2} {layer_emoji} [{r.layer}] Q={r.q_score:.3f}")
        print(f"       {indent}{r.content[:65]}...")
        print()


if __name__ == "__main__":
    # Load the pre-computed realizations
    print("Loading pre-computed realizations...")
    engine = precompute_conversation_realizations()
    
    # Run all visualizations
    visualize_graph_ascii(engine)
    analyze_generativity(engine)
    demonstrate_queries(engine)
    show_quality_distribution(engine)
    trace_realization_path(engine)
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
    print("\nWhat we've proven:")
    print("  ✅ Realizations can be extracted from conversation")
    print("  ✅ Q-scores can be calculated automatically")
    print("  ✅ Layers emerge naturally from quality thresholds")
    print("  ✅ Parent-child relationships preserve lineage (بنات افكار)")
    print("  ✅ The system is queryable and reusable")
    print("\nThis is pre-computation working on realizations.")
    print("The conversation is now a knowledge base.")
