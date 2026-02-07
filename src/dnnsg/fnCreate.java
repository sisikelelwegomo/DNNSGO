package dnnsg;

import java.io.PrintWriter;
import java.util.List;
import java.util.Random;

/**
 *
 * @author Clint van Alten
 */
public class fnCreate {
   
    
    private static final Random rand = new Random();
    
    
    public static void createNetwork(network Net, int[] S, double skipPercentage, PrintWriter writer) {
    // Create input and output layers first
    createOuterLayers(Net, S[0], S[S.length-1]);
    
    // Insert hidden layers
    layernode current = Net.inputlayernode;
    for (int i = 1; i < S.length-1; i++) {
        insertLayer(current, S[i], skipPercentage, writer);
        current = current.next;
    }
    
    // Connect all layers properly
    connectAllLayers(Net, skipPercentage);
    
    logNetworkStructure(Net, writer);
}
    
    
public static void verifyOutputLayerConnections(network Net, PrintWriter writer) {
    if (Net == null) {
        writer.println("ERROR: Network is null");
        return;
    }
    
    if (Net.outputlayernode == null) {
        writer.println("ERROR: Output layer is null");
        return;
    }
    
    layernode outputLayer = Net.outputlayernode;
    
    // Check if output layer has nodes
    if (outputLayer.firstnode == null) {
        writer.println("ERROR: Output layer has no nodes");
        return;
    }
    
    layernode prevLayer = outputLayer.prev;
    
    if (prevLayer == null) {
        writer.println("Warning: Output layer has no previous layer!");
        return;
    }
    
    // Check if previous layer has nodes
    if (prevLayer.firstnode == null) {
        writer.println("ERROR: Previous layer has no nodes");
        return;
    }
    
    int connectionCount = 0;
    node srcNode = prevLayer.firstnode;
    while (srcNode != null) {
        edge e = srcNode.firstedge;
        while (e != null) {
            if (e.target != null && e.target.layerindex == outputLayer.getIndex()) {
                connectionCount++;
            }
            e = e.next;
        }
        srcNode = srcNode.next;
    }
    
    int expectedConnections = prevLayer.getNodeCount() * outputLayer.getNodeCount();
    writer.println("Output layer connections: " + connectionCount + "/" + expectedConnections);
    
    if (connectionCount == 0) {
        writer.println("ERROR: No connections to output layer detected!");
    } else if (connectionCount < expectedConnections) {
        writer.println("Warning: Partial connections to output layer");
    }
}
    
    
    
    public static void connectAllLayers(network Net, double skipPercentage) {
    layernode layer1 = Net.inputlayernode;
    
    while (layer1 != null) {
        layernode layer2 = layer1.next;
        while (layer2 != null) {
            // Connect each node in layer1 to each node in layer2
            for (node src = layer1.firstnode; src != null; src = src.next) {
                for (node dest = layer2.firstnode; dest != null; dest = dest.next) {
                    if (rand.nextDouble() < skipPercentage || layer2 == layer1.next) {
                        edge e = new edge(src, dest);
                        e.next = src.firstedge;
                        src.firstedge = e;
                    }
                }
            }
            layer2 = layer2.next;
        }
        layer1 = layer1.next;
    }
}
    
    
    public static void adjustSkipConnections(network Net, double skipPercentage, PrintWriter writer) {
            
    layernode x = Net.inputlayernode;
    
    while (x != null) {
        
        layernode y  = x.next;
        
        if (y != null) {
            
            insertSkipConnections(x, y, skipPercentage, null, writer);
            
        }
        
        x = x.next;
    }
}
        
    public static int insertSkipConnections(layernode x, layernode y, double skipPercentage, List<edge> activeSkipConnections, PrintWriter writer) {
        
    int skipConnectionsAdded = 0; 
    
    int totalNodes = x.getNodeCount();

    for (node n = x.firstnode; n != null; n = n.next) {
        
        for (node m = y.firstnode; m != null; m = m.next) {
            
            if (shouldAddSkipConnection(n, m, skipPercentage, totalNodes)) {
                
                edge e = new edge(n, m);
                
                e.next = n.firstedge;        
                
                n.firstedge = e;

                if (activeSkipConnections != null) {
                    
                    activeSkipConnections.add(e); 
                    
                }
                
                skipConnectionsAdded++;      
            }
        }
    }

    writer.println("Skip Connections Updated: Total = " + skipConnectionsAdded);
    
    return skipConnectionsAdded;
}
        
    public static boolean shouldAddSkipConnection(node n, node m, double skipPercentage, int totalNodes) {
        
        if (skipPercentage == 0.0) return false; 
        
        if (skipPercentage == 1.0) return true;  

        int threshold = (int) (skipPercentage * totalNodes);
        
        return n.nodeindex < threshold; 
    }
    
    
    public static void createOuterLayers(network Net, int Inputsize, int Outputsize) {
    
        layernode x = new layernode();
        
        Net.inputlayernode = x;
        
        createNodeList(x, Inputsize, 0);
                                        
        layernode y = new layernode();
        
        Net.outputlayernode = y;
        
        createNodeList(y, Outputsize, 2); 
        
        x.next = y;
        
        y.prev = x;
        
        connectLayers(x,y);    
    }
    
    
    public static void createNodeList(layernode x, int k, int t) {
        
/*  creates a layer containing k nodes of type t,
    if t = -1, then each node get a randomly chosen activation function
    if t = 1, then each node gets relu activation function
*/    

        node n = new node(t);
        
        x.firstnode = n;
        
        for (int i = 1; i < k; i++) {
            
            node m = new node(t);
            
            n.next = m;
            
            n = m;
        }
    }



    
public static void insertLayer(layernode prevLayer, int size, double skipPercentage, PrintWriter writer) {
    layernode newLayer = new layernode();
    
    // Insert into the network chain
    newLayer.next = prevLayer.next;
    newLayer.prev = prevLayer;
    if (prevLayer.next != null) {
        prevLayer.next.prev = newLayer;
    }
    prevLayer.next = newLayer;
    
    // Create nodes with tanh activation
    createNodeList(newLayer, size, 3); // 3 = tanh activation
}


    
    /**
 * Connects each node in the fromLayer to each node in the toLayer without skip logic.
 * This is used for Student network connections.
     * @param fromLayer
     * @param toLayer
     * @param writer
 */
public static void connectLayers(layernode fromLayer, layernode toLayer) {
    for (node n = fromLayer.firstnode; n != null; n = n.next) {
        
        for (node m = toLayer.firstnode; m != null; m = m.next) {
            
            edge e = new edge(n, m); // Regular connection
            e.next = n.firstedge;    // Add edge to the source node's edge list
            n.firstedge = e;
        }
    }

}


/**
 * Connects each node in layer x to all nodes in layer y using skipPercentage logic.
 * This is used for Teacher network connections.
     * @param x
     * @param y
     * @param skipPercentage
     * @param writer
 */

public static void connectLayers(layernode x, layernode y, double skipPercentage) {
    node n = x.firstnode;
    while (n != null) {
        connectNodeToLayer(n, y, skipPercentage);
        n = n.next;
    }
}



    
    public static void connectNodeToLayer(node n, layernode y, double skipPercentage) {
        
    // adds an edge from node n to every node in layer y.
    
            node m = y.firstnode;
        
        while (m != null) {
            
            if (rand.nextDouble() < skipPercentage){
        
            edge e = new edge();
            
            e.target = m;
            
            e.next = n.firstedge;
            
            n.firstedge = e;
            
                }
            
            m = m.next;        
            }
        } 
        

    
    
        public static void connectNodeToLayer(node n, layernode y) {
        
    // adds an edge from node n to every node in layer y.
    
        node m = y.firstnode;
        
        while (m != null) {
        
            edge e = new edge();
            
            e.target = m;
            
            e.next = n.firstedge;
            
            n.firstedge = e;
            
            m = m.next;        
        } 
        
    }
       
        
    
    public static void connectLayerToNode(layernode y, node n) {
    
    // adds an edge from every node in layer y to node n 
        
        node m = y.firstnode;
            
        while (m != null) {
            
            edge e = new edge();
            
            e.target = n;
            
            e.next = m.firstedge;
            
            m.firstedge = e;
                
            m = m.next;
        }
    }
    
    
    
    public static void insertNode(layernode x, int t, double skipPercentage) {
        
/*  inserts a new node in layer x and connects it to every node in every 
    subsequent layer and connects every node in every previous layer to it
*/
        
        node n = new node(t);
        
        n.next = x.firstnode;
        
        x.firstnode = n;
        
        layernode y = x.next;
        
        while (y != null)  {
        
            connectNodeToLayer(n, y, skipPercentage);
            
            y = y.next;
        }
        
        y = x.prev;
        
        while (y != null)  {
        
            connectLayerToNode(y, n);
            
            y = y.prev;
        }
    }

    
    public static void createNetworkStd(network Net, int[] S) {
    
/*  creates a standard network with layers based on S,
    where consecutive layers are connected
    S[0] is the number of input nodes,
    S[S.length - 1] is the number of output nodes.
    The network is created by first creating all layers and then
    connecting the consecutive layers.
*/

        layernode x = new layernode();
        
        Net.inputlayernode = x;
        
        createNodeList(x, S[0], 0);        
        
        for (int i = 1; i < S.length; i++) {
        
            layernode z = new layernode();
    
            x.next = z;
        
            z.prev = x;
            
            if (i == S.length-1) { createNodeList(z, S[i], 2);}
                
            else createNodeList(z, S[i], 3);
    
            connectLayers(x, z);

            x = z;
        
        }
        
        Net.outputlayernode = x;

    }
    
    
    
public static void logNetworkStructure(network Net, PrintWriter writer) {
    layernode x = Net.inputlayernode;
    writer.println("\nNetwork Structure:");
    int layerIndex = 0;

    while (x != null) {
        int nodeCount = x.getNodeCount();
        int edgeCount;

        // Determine edge count based on layer type
        if (x == Net.outputlayernode) {
            // For output layer, count incoming edges from previous layer
            edgeCount = (x.prev != null) ? x.prev.getNodeCount() * nodeCount : 0;
        } else {
            // For other layers, count outgoing edges to next layer
            edgeCount = (x.next != null) ? x.getNodeCount() * x.next.getNodeCount() : 0;
        }

        // Determine layer type label
        String layerType;
        if (layerIndex == 0) {
            layerType = "Input Layer";
        } else if (x == Net.outputlayernode) {
            layerType = "Output Layer";
        } else {
            layerType = "Hidden Layer " + layerIndex;
        }

        writer.println(layerType + ": " + nodeCount + " nodes, " + edgeCount + " edges.");
        
        x = x.next;
        layerIndex++;
    }
    writer.println();
}




   
}