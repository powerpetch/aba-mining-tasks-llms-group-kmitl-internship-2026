// Cytoscape-based rendering, replacing mu_team's hand-rolled SVG renderer. Two explicit,
// separately-labeled views (ABA vs AA) instead of one conflated ad hoc tree — see aba_builder.py
// / pyarg_service.py docstrings for why that conflation was the actual bug being fixed here.

const ROLE_COLORS = {
  head: "#d9a441",
  assumption_pos: "#3fb37f",
  assumption_neg: "#e0575b",
  contrary: "#4f5561",
};

const STYLESHEET = [
  {
    selector: "node",
    style: {
      "background-color": "#4f8cff",
      label: "data(label)",
      color: "#e6e8eb",
      "font-size": 9,
      "text-wrap": "wrap",
      "text-max-width": "90px",
      width: 26,
      height: 26,
      "text-valign": "bottom",
      "text-margin-y": 4,
    },
  },
  { selector: "node[role='head']", style: { "background-color": ROLE_COLORS.head, shape: "round-rectangle", width: 34, height: 34 } },
  { selector: "node[role='assumption_pos']", style: { "background-color": ROLE_COLORS.assumption_pos } },
  { selector: "node[role='assumption_neg']", style: { "background-color": ROLE_COLORS.assumption_neg } },
  { selector: "node[role='contrary']", style: { "background-color": ROLE_COLORS.contrary, shape: "diamond", "border-width": 2, "border-color": "#9aa1ac" } },
  { selector: "node[role='argument']", style: { "background-color": "#4f8cff", shape: "ellipse", width: 30, height: 30 } },
  {
    selector: "edge",
    style: {
      width: 1.5,
      "line-color": "#5a6070",
      "target-arrow-color": "#5a6070",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      opacity: 0.85,
    },
  },
  { selector: "edge[kind='support']", style: { "line-color": "#3fb37f", "target-arrow-color": "#3fb37f" } },
  { selector: "edge[kind='attack']", style: { "line-color": "#e0575b", "target-arrow-color": "#e0575b", "line-style": "solid" } },
  { selector: "edge[kind='defeat']", style: { "line-color": "#e0575b", "target-arrow-color": "#e0575b" } },
];

let cy = null;

function ensureCy() {
  if (cy) return cy;
  cy = cytoscape({
    container: document.getElementById("cy"),
    style: STYLESHEET,
    wheelSensitivity: 0.2,
  });
  return cy;
}

/** ABA view: every assumption, contrary atom, and head atom in the framework as a node;
 * support rules (body -> head) and attack-derivation rules (attacker -> contrary) as edges.
 * This is a 1:1 rendering of framework.rules / framework.contraries — nothing summarized,
 * nothing hidden (the bug this app exists to avoid). */
function renderAbaView(data) {
  const c = ensureCy();
  const nodes = [];
  const seen = new Set();
  const addNode = (id, role) => {
    if (seen.has(id)) return;
    seen.add(id);
    nodes.push({ data: { id, label: id, role } });
  };

  for (const atom of data.aba.language) {
    addNode(atom, data.aba.atom_roles[atom] || "atom");
  }

  const edges = [];
  for (const rule of data.aba.rules) {
    const isAttackRule = rule.id.startsWith("r_attack_");
    for (const premise of rule.body) {
      edges.push({
        data: {
          id: `${rule.id}_${premise}`,
          source: premise,
          target: rule.head,
          kind: isAttackRule ? "attack" : "support",
        },
      });
    }
  }

  c.elements().remove();
  c.add([...nodes, ...edges]);
  // breadthfirst rooted at the head nodes doesn't work here: every edge points INTO a head
  // or contrary atom, never out of one, so breadthfirst can't expand a hierarchy from them —
  // it was dumping almost every node into one overlapping row. cose (force-directed) handles
  // this fan-in shape correctly regardless of edge direction.
  c.layout({
    name: "cose",
    animate: false,
    nodeRepulsion: 25000,
    idealEdgeLength: 120,
    nodeOverlap: 20,
    componentSpacing: 100,
  }).run();
}

/** AA view: the real Dung instantiation from framework.generate_af() — constructed arguments
 * as nodes, derived defeats as edges. Genuinely different objects from the ABA view above,
 * not a relabeling of the same heuristic tree. */
function renderAaView(data) {
  const c = ensureCy();
  const nodes = data.aa.arguments.map((arg) => ({
    data: {
      id: arg.id,
      label: arg.conclusion,
      role: "argument",
    },
  }));
  const edges = data.aa.defeats.map((d, i) => ({
    data: { id: `defeat_${i}`, source: d.source, target: d.target, kind: "defeat" },
  }));

  c.elements().remove();
  c.add([...nodes, ...edges]);
  c.layout({
    name: "cose",
    animate: false,
    nodeRepulsion: 25000,
    idealEdgeLength: 120,
    nodeOverlap: 20,
    componentSpacing: 100,
  }).run();
}
