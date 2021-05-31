from networkx.algorithms import centrality
from apollo.utils.util import Util
from apollo.utils.scraper import Scraper
from apollo.utils.graph import Graph
from apollo.utils.GCN import GCN_Mult
from apollo.utils.processor import Processor
from apollo.utils.scraper import Scraper
from apollo.utils.data_manipulation import Manipulator
import pandas as pd

s = Scraper(["AAPL", "MSFT","MMM","AXP","AMGN","CAT","GS","HD","TRV","JPM","WMT","CVX","INTC"]) #data scraper object
p = Manipulator #data processor
ut = Util() #utility package
g = Graph() #graph object
s.set_period("7d")
s.set_date_range(["02-01-2015","02-12-2015"])

##fetch news and finaicial data
#s.check_status()
#news = s.fetch_news_data(10)
#Util.print_to_csv(news,"news.csv","data")
[price,vol] = s.fetch_financial_data()

#p.set_data(pd.read_csv("/home/jay/apollo/data/news.csv"))
#[df_ner,df_links]=p.process_news_data()
df_links = pd.read_csv("/home/jay/apollo/data/df_links_2011_2015.csv")
comention_matrix = p.news_to_correlation_matrix(df_links)
df_links = p.subset_comention_links(df_links,5)


g.set_data([df_links])
graphs=g.create_graphs()
g.visualize_graphs()
centrality = g.get_centralities()
g.visualize_centralities_barplot(centrality)
[cliques, optimal_clique]=g.find_optimal_number_of_cliques()
clx=g.find_centralities_in_cliques(cliques)
g.plot_cliques(clx)




#print to csv
#ut.print_to_csv(price,"price.csv")
#ut.print_to_csv(vol,"vol.csv")