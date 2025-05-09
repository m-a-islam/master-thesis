# plot_genotype.py
# Visualizes a Genotype object using graphviz for CNN cells.

import sys
import os
import genotypes
from graphviz import Digraph

def plot(genotype, filename_suffix, genotype_name=""):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot'
    )
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # Create a more descriptive filename that includes the architecture name
    full_filename = f"{genotype_name}_{filename_suffix}" if genotype_name else filename_suffix
    
    try:
        g.render(full_filename, view=True)
    except Exception as e:
        print(f"Error rendering graph: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"usage:\n python {sys.argv[0]} ARCH_NAME")
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval(f'genotypes.{genotype_name}')
    except AttributeError:
        print(f"{genotype_name} is not specified in genotypes.py")
        sys.exit(1)

    # Pass the genotype_name to the plot function
    plot(genotype.normal, "normal", genotype_name)
    plot(genotype.reduce, "reduction", genotype_name)
