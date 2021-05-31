import torch
import torchvision 
import torch.nn.functional as F
import torch_geometric.data as tgd
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from sklearn.manifold import TSNE
from IPython.display import Javascript  # Restrict height of output cell.
from sklearn.model_selection import ShuffleSplit
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))


class GCN_Mult(torch.nn.Module):
    
    def __init__(self, hidden_channels, num_feats,seeds=12345):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        super(GCN_Mult, self).__init__()
        torch.manual_seed(seeds)
        num_labels=2
        self.conv1 = GCNConv(num_feats, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_labels)

    def forward(self, x, edge_index):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

