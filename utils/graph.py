import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import statistics
import louvain
import igraph as ig
import leidenalg as la
from collections import Counter
from apollo.utils.util import Util
from apollo.utils.data_manipulation import Manipulator
from py3plex.visualization import colors
from apollo.utils.LayeredNetworkGraph import LayeredNetworkGraph
from apollo.utils.processor import Processor
from py3plex.core import multinet
from networkx.algorithms.community.kclique import k_clique_communities
from py3plex.algorithms.community_detection import community_wrapper as cw
from tqdm import tqdm
tqdm.pandas()
ut = Util()


col_pal = {0: '#F1E8F3',1: '#A8DDFF',2: '#FF8A5B',3: '#74D3AE',4: '#93B7BE',5: '#D1B1CB',6: '#BAF2BB',7: '#FFA69E',8: '#97EAD2',9: '#34E4EA',10: '#B95F89',99: '#828A95'}
sns.set(rc={'figure.figsize': (20, 15)})


class Graph(object):

    def __init__(self, data_array=[pd.DataFrame()]):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        self.data_array = data_array
        self.data_num = len(data_array)
        self.graphs = []
        self.unique_nodes = []
        self.multiplex_graph = multinet.multi_layer_network(network_type="multiplex")

    def create_graphs(self):
        ''' Function to create graphs from self.graphs(can be more than one graph) and stores the graphs inside the self.graphs 
        '''
        for i in range(self.data_num):
            self.data_array[i].reset_index(inplace=True, drop=True)
            graph = nx.Graph()
            if not self.unique_nodes:
                self.unique_nodes = self.pull_unique_nodes()
            for nd in self.unique_nodes:
                graph.add_node(nd)
            for link in tqdm(self.data_array[i].index):
                graph.add_edge(self.data_array[i].iloc[link]['from'], self.data_array[i].iloc[link]
                               ['to'], weight=self.data_array[i].iloc[link]['weight'])
            self.graphs.append(graph)
            print(graph)
            print("Graph created ...")
        return self.graphs

    def get_graphs(self):
        ''' Function to return created graphs

            :return grahs: List of graphs
        '''
        if not self.graphs:
            print("No graphs created. Create Graphs first")
            return
        else:
            return self.graphs
    
    def get_multiplex_graph(self):
        return self.multiplex_graph

    def pull_unique_nodes(self):
        ''' Function to extract unique nodes from data

            :return unique_nodes: List of unique nodees
        '''
        if not self.data_array:
            print("No data loaded.")
            return
        unique_nodes = []
        for row in self.data_array[0].iterrows():
            if row[1]["from"] not in unique_nodes:
                unique_nodes.append(row[1]["from"])
            if row[1]["to"] not in unique_nodes:
                unique_nodes.append(row[1]["to"])
        return unique_nodes

    def visualize_graphs(self):
        ''' Function to visualize graphs using the Kamada Kawai Layout
        '''
        for G in self.graphs:
            pos = nx.kamada_kawai_layout(G)
            nodes = G.nodes()
            fig, axs = plt.subplots(1, 1, figsize=(15, 20))
            el = nx.draw_networkx_edges(G, pos, alpha=0.1, ax=axs)
            nl = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='#FF427b',
                                        node_size=50, ax=axs)
            ll = nx.draw_networkx_labels(
                G, pos, font_size=10, font_family='sans-serif')
            plt.show()

    def generate_graph_labels(self):
        ''' Function to generate various labels and nodes for the multilayer graph use

            :return :
        '''
        node_labels = {}
        nodes_multi_layer = {}
        type_count = 0
        node_count = 0
        for G in self.graphs:
            node_type = "t"+(type_count+1)
            for node in G.nodes():
                # set the node name as the key and the label as its value
                node_labels[node] = node

                # create nodes for multilayered graph
                nodes_multi_layer[node_count] = {
                    "node": node, "type": node_type}

                node_count += 1
            type_count += 1
        return [node_labels, nodes_multi_layer]

    def get_centralities(self):
        ''' Function to generate various labels and nodes for the multilayer graph use

            :return :
        '''
        centralities = []
        for G in self.graphs:
            nodes = []
            eigenvector_cents = []
            ec_dict = nx.eigenvector_centrality(
                G, max_iter=1000, weight='weight')
            for node in tqdm(G.nodes()):
                nodes.append(node)
                eigenvector_cents.append(ec_dict[node])

            df_centralities = pd.DataFrame(data={'entity': nodes,
                                                 'eigenvector': eigenvector_cents})

            centralities.append(df_centralities)
        return centralities

    def visualize_centralities_barplot(self, centralities, filter=20, save_figure=False):
        ''' Function to generate various labels and nodes for the multilayer graph use

            :param centralities: list of centralities of each graph
            :param filter: number top records to filter on
            :param save_figure: boolean to export the image or not
        '''
        count = 1
        for centrality in centralities:
            df_cent_top = centrality.sort_values(
                'eigenvector', ascending=False).head(filter)
            df_cent_top.reset_index(inplace=True, drop=True)
            fig, axs = plt.subplots(figsize=(10, 7))
            g = sns.barplot(data=df_cent_top,
                            x='eigenvector',
                            y='entity',
                            dodge=False,
                            orient='h',
                            hue='eigenvector',
                            palette='viridis',)

            g.set_yticks([])
            g.set_title('Most influential entities in network Graph '+ str(count))
            g.set_xlabel('Eigenvector centrality')
            g.set_ylabel('')
            g.set_xlim(0, max(df_cent_top['eigenvector'])+0.1)
            g.legend_.remove()
            g.tick_params(labelsize=5)
          

            for i in df_cent_top.index:
                g.text(df_cent_top.iloc[i]['eigenvector'] +
                       0.005, i+0.25, df_cent_top.iloc[i]['entity'])
            plt.show()
            ut.save_figure(g, 'cent_plot.png')
            nodes = []
            eigenvector_cents = []
            count += 1

    def find_optimal_number_of_cliques(self, draw=False):
        ''' Function to generate various labels and nodes for the multilayer graph use

            :param centralities: list of centralities of each graph
            :param filter: number top records to filter on
            :param save_figure: boolean to export the image or not
        '''
        clique_sizes = range(2, 30)
        cliques = []
        num = 0
        optimal_clique = []  # will hold the optimal clique size for each graph
        for G in self.graphs:
            n_cliques = []
            for k in tqdm(clique_sizes):
                n_cliques.append(len(list(k_clique_communities(G, k))))

            # clique sizes should be greater than one, hence least clique size is 2
            optimal_clique.append(2+(n_cliques.index(max(n_cliques))))
            print(n_cliques)

            if(draw):
                df_relplot = pd.DataFrame(
                    data={'k': clique_sizes, 'n': n_cliques})
                sns.relplot(data=df_relplot, x='k', y='n')
            cliques.append(list(k_clique_communities(G, optimal_clique[num])))
            num += 1

        return [cliques, optimal_clique]

    def find_centralities_in_cliques(self, cliques):
        ''' Function to generate various labels and nodes for the multilayer graph use

            :param centralities: list of centralities of each graph
            :param filter: number top records to filter on
            :param save_figure: boolean to export the image or not
        '''
        clx = []
        num = 0
        for G in self.graphs:
            eigenvector_cents = []
            entities = []
            clique_ids = []
            for id, clique in enumerate(cliques[num]):
                sg = G.subgraph(list(clique))

                nodes = sg.nodes()

                clique_ids.extend(np.repeat(id, len(nodes)))
                entities.extend(nodes)

                ec_dict = nx.eigenvector_centrality(sg, max_iter=1000, weight='weight')

                for entity in nodes:
                    eigenvector_cents.append(ec_dict[entity])
            df_cliques = pd.DataFrame(data={
                'clique': clique_ids,
                'entity': entities,
                'centrality': eigenvector_cents
            })
            clx.append(df_cliques)
            num += 1
        return clx

    def plot_cliques(self, clx):
        ''' Function to generate various labels and nodes for the multilayer graph use

            :param centralities: list of centralities of each graph
            :param filter: number top records to filter on
            :param save_figure: boolean to export the image or not
        '''
        clique_num=0
        
        for G in self.graphs:

            df_cliques = clx[clique_num]
            G_clique = G.subgraph(df_cliques['entity'].unique())
            pos = nx.kamada_kawai_layout(G_clique)
            nodes = G_clique.nodes()
            if(len(df_cliques['clique'].unique())>1):
              fig, axs = plt.subplots(max(df_cliques['clique'])+1, 2, figsize=(15,40))

            

            for clique in range(max(df_cliques['clique'])+1):
                if(len(df_cliques['clique'].unique())<2):
                    print("Clique Number Must Be Greater Than 2")
                    break
                node_colors = [col_pal[clique] if node in df_cliques[df_cliques['clique']
                                                                     == clique]['entity'].values else col_pal[99] for node in nodes]
                sizes = [40 if node in df_cliques[df_cliques['clique']
                                                  == clique]['entity'].values else 15 for node in nodes]
                edge_colors = ['black' if node in df_cliques[df_cliques['clique']
                                                             == clique]['entity'].values else col_pal[99] for node in nodes]

                ec = nx.draw_networkx_edges(G_clique, pos, alpha=0.05, ax=axs[clique, 0])
                nc = nx.draw_networkx_nodes(G_clique, pos, nodelist=nodes, node_color=node_colors,
                                            node_size=sizes, ax=axs[clique, 0],
                                            edgecolors=edge_colors)

                df_clique_ind = df_cliques[df_cliques['clique'] == clique]
                df_clique_ind = df_clique_ind.sort_values(
                    'centrality', ascending=False).head(15)
                df_clique_ind.reset_index(inplace=True, drop=True)

                g = sns.barplot(data=df_clique_ind,
                                x='centrality',
                                y='entity',
                                hue='clique',
                                palette=col_pal,
                                dodge=False,
                                orient='h',
                                ax=axs[clique, 1])

                g.set_yticks([])
                g.set_title(f'Clique {clique}')
                g.set_xlabel('')
                g.set_ylabel('')
                g.legend_.remove()
                g.tick_params(labelsize=5)
               

                for i in df_clique_ind.index:
                    g.text(max(df_clique_ind['centrality'])/30,
                           i+0.15, df_clique_ind.iloc[i]['entity'])
                
                plt.show()

            clique_num = clique_num+1
            sns.despine()

    def set_data(self,data_array):
        ''' Initialization function for named entity recognition parts

            :param data_array: Data array containing dataframes to be graphed with the structure [from,to, weight]
        '''
        self.data_array = data_array
        self.data_num = len(data_array)


    def create_multiplex_graph(self,draw=False):
        ''' Initialization function for named entity recognition parts

            :param data_array: Data array containing dataframes to be graphed with the structure [from,to, weight]
        '''
        type_num=0
        #edge_type=["co-mention","price/vol"]
        unique_nodes = self.pull_unique_nodes()
        for i in range(self.data_num):
            for nd in unique_nodes:
               self.multiplex_graph.add_nodes({"source": nd, "type": ('t'+type_num)})
            for link in tqdm(self.data_array[i].index):
                self.multiplex_graph.add_edges({"source":self.data_array[i].iloc[link]['from'], "layer":type_num, "target":self.data_array[i].iloc[link]['to'],"type":('t'+type_num),
                        "source_type":('t'+type_num), "weight":self.data_array[i].iloc[link]['weight'],"target_type":('t'+type_num)})
            type_num+=1

        #add edges between layers for common nodes in both layers
        #have to revise this line of code
        for n in unique_nodes:
            self.multiplex_graph.add_edges({"source":n,"source_type":"t1","target":n, "target_type":"t2"})
    
    def visualize_multiplex_graph(self):
        ''' Initialization function for named entity recognition parts

            :param data_array: Data array containing dataframes to be graphed with the structure [from,to, weight]
        '''
        fig = plt.figure(figsize=(15,20))
        ax = fig.add_subplot(111, projection='3d')
        [node_labels, nodes_multi_layer]=self.generate_graph_labels()
        LayeredNetworkGraph(self.graphs, node_labels=node_labels ,ax=ax, layout=nx.spring_layout)
        ax.set_axis_off()
        plt.show()
        ut.save_figure(fig, 'multilayered.png')
    
    def multiplex_community_detection(self):
        ''' Initialization function for named entity recognition parts

            :param data_array: Data array containing dataframes to be graphed with the structure [from,to, weight]
        '''
        partitions = cw.louvain_communities(self.multiplex_graph)
        dict(sorted(partitions.items(), key=lambda item: item[1]))
        return partitions
    
    
    def multiplex_visualize_communities(self,partition):
        ''' Initialization function for named entity recognition parts

            :param data_array: Data array containing dataframes to be graphed with the structure [from,to, weight]
        '''
        top_n = 10
        partition_counts = dict(Counter(partition.values()))
        top_n_communities = list(partition_counts.keys())[0:top_n]

        color_mappings = dict(zip(top_n_communities,[x for x in colors.colors_default if x !="black"][0:top_n]))
        network_colors = [color_mappings[partition[x]] if partition[x] in top_n_communities else "black" for x in self.multiplex_graph.get_nodes()]
        nds =list(self.multiplex_graph.get_nodes())
        #nds = [row[0] for row in nds]
        #print(nds)
        print(self.multiplex_graph.core_network)
        print(network_colors)
        #print(A.core_network)

        # visualize the network's communities
        self.visualize_partitions(self.multiplex_graph.core_network,
                            color_list=network_colors,
                            layout_parameters={"iterations": 200},
                            scale_by_size=True,
                            node_sizes= 50,
                            labels=nds,
                            edge_width=0.2,
                            label_font_size=10,
                            layout_algorithm="force",
                            legend=False)
        plt.show()
    
    
    def visualize_partitions(self,g,color_list=None,display=False,node_size=1,text_color="black",node_sizes=None,layout_parameters=None,legend=None,scale_by_size=True,layout_algorithm="force",edge_width=0.01,alpha_channel=0.5,labels=None,draw=True,label_font_size=2):
        ''' Initialization function for named entity recognition parts

            :param data_array: Data array containing dataframes to be graphed with the structure [from,to, weight]
        '''
        print("Beginning parsing..")
        nodes = g.nodes(data=True)
        potlabs = []
        #    fig, ax = plt.subplots()
        for node in nodes:
            try:
                potlabs.append(node[0][1])
            except:
                potlabs.append("unlabeled")

        if color_list is None:
            unique_colors = np.unique(potlabs)
            color_mapping = dict(zip(list(unique_colors), colors.colors_default))
            try:
                color_list = [color_mapping[n[1]['type']] for n in nodes]
            except:
                print("Assigning colors..")
                color_list = [1] * len(nodes)

        
        node_types = [x[1] for x in g.nodes()]
        assert len(node_types) == len(color_list)

        try:
            cols = color_list            
        except Exception as es:
            print("Using default palette")
            cols = colors.colors_default            
        id_col_map = {}
        for enx, j in enumerate(set(color_list)):
            id_col_map[j] = j
        id_type_map = dict(zip(color_list, node_types))
        final_color_mapping = [id_col_map[j] for j in color_list]
        color_to_type_map = {}
        for k, v in id_type_map.items():
            actual_color = id_col_map[k]
            color_to_type_map[actual_color] = id_type_map[k]

        degrees = dict(nx.degree(nx.Graph(g)))

        if scale_by_size:
            nsizes = [
                np.log(v) * node_size if v > 10 else v for v in degrees.values()
            ]
        else:
            nsizes = [node_size for x in g.nodes()]

        if not node_sizes is None:
            nsizes = node_sizes

        # standard force -- directed layout
        if layout_algorithm == "force":
            pos = multinet.compute_force_directed_layout(g, layout_parameters)

        # random layout -- used for initialization of more complex algorithms
        elif layout_algorithm == "random":
            pos = multinet.compute_random_layout(g)

        elif layout_algorithm == "custom_coordinates":
            pos = layout_parameters['pos']

        elif layout_algorithm == "custom_coordinates_initial_force":
            pos = multinet.compute_force_directed_layout(g, layout_parameters)
        else:
            raise ValueError('Uknown layout algorithm: ' + str(layout_algorithm))

        if draw:
            nx.draw_networkx_edges(g,
                                pos,
                                alpha=alpha_channel,
                                edge_color="black",
                                width=edge_width,
                                arrows=False)
            scatter = nx.draw_networkx_nodes(g,
                                            pos,
                                            nodelist=[n1[0] for n1 in nodes],
                                            node_color=final_color_mapping,
                                            node_size=nsizes,
                                            alpha=alpha_channel)
        if labels is not None:
            for el in labels:
                pos_el = pos[el]
                if draw:
                    plt.text(pos_el[0],
                            pos_el[1],
                            el,
                            fontsize=label_font_size,
                            color=text_color)


    #        nx.draw_networkx_labels(g, pos, font_size=label_font_size)

        plt.axis('off')

        #  add legend {"color":"string"}
        if legend is not None and legend:
            legend_colors = set(id_col_map.values())
            if len(legend_colors) > 6:
                fs = "small"
            else:
                fs = "medium"
            markers = [
                plt.Line2D([0, 0], [0, 0], color=key, marker='o', linestyle='')
                for key in legend_colors
            ]
            if draw:
                plt.legend(markers,
                        [color_to_type_map[color] for color in legend_colors],
                        numpoints=1,
                        fontsize=fs)

        if display:
            plt.show()

        if not draw:
            return g, nsizes, final_color_mapping, pos
        
                  


if __name__ == "__main__":
    p = Manipulator()
    df_links = pd.read_csv("/home/jay/apollo/data/df_links.csv")
    comention_matrix = p.news_to_correlation_matrix(df_links)
    df_links = p.subset_comention_links(df_links,5)
    g = Graph([df_links])

    graphs=g.create_graphs()
    g.visualize_graphs()
    centrality = g.get_centralities()
    g.visualize_centralities_barplot(centrality)
    [cliques, optimal_clique]=g.find_optimal_number_of_cliques()
  
    clx=g.find_centralities_in_cliques(cliques)
   
    g.plot_cliques(clx)


    # [ner,links]=n.processfile()

    # n.print_to_csv(links,"df_links.csv")
    # n.print_to_csv(ner,"df_ner.csv")
