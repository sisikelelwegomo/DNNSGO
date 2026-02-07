package dnnsg;

import java.io.PrintWriter;

/**
 *
 * @author Clint van Alten
 * @author Sisikelelwe Gomo
 */
public class fnAdd {
    

    public static void networkAddup(network Net, PrintWriter writer) {
    
        addEdgesByWeight(Net, writer);
    
        addEdgesByLoss(Net, writer);
    
        addEdgesByAccuracy(Net, writer);
        addNodesByAccuracy(Net, writer);
    
        addNodes(Net, writer);
    
        reinforceEdges(Net, writer);
    
        addEdgesToIsolatedNodes(Net, writer);
    
        populateEmptyLayers(Net, writer);
    }

    
    
    public static void addEdgesByWeight(network Net, PrintWriter writer) {
    
        layernode x = Net.outputlayernode.prev;
    
        node n;
    
        edge newEdge;
    
        double minweight = 0.00001;
    
        int edgesAdded = 0;

        while (x != null) {
        
            n = x.firstnode;
        
            while (n != null) {
            
                // Ensure each node has at least one outgoing edge
            
                if (n.firstedge == null || Math.abs(n.firstedge.weight) < minweight) {
                
                    newEdge = new edge();
                
                    newEdge.weight = minweight;
                
                    // Connect to a node in the next layer if possible
                
                    layernode targetLayer = (x.next != null) ? x.next : Net.outputlayernode;
                
                    if (targetLayer.firstnode != null) {
                    
                        newEdge.target = targetLayer.firstnode;
                    
                        newEdge.next = n.firstedge;
                    
                        n.firstedge = newEdge;
                    
                        edgesAdded++;
                    }
                
                }
            
                n = n.next;
            }
        
            x = x.prev;
        }

        System.out.println("Number of edges added by weight: " + edgesAdded);
        
        writer.println("Number of edges added by weight: " + edgesAdded);
    }

    public static void addEdgesByLoss(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next;
    
        int edgesAdded = 0;
    
        double newEdgeWeight = 0.005;

        double currentLoss = fnTrain.lastComputedLoss;
    
        double lossThreshold = 3.0;

        if (currentLoss > lossThreshold) {
        
            while (x != null && x.next != null) {
            
                node n = x.firstnode;
            
                while (n != null) {
                
                    edge newEdge = new edge();
                
                    newEdge.weight = newEdgeWeight;

                    if (x.next.firstnode != null) {
                    
                        newEdge.target = x.next.firstnode;
                    
                        newEdge.next = n.firstedge;
                    
                        n.firstedge = newEdge;
                    
                        edgesAdded++;
                    
                    }
                
                    n = n.next;
                
                }
            
                x = x.next;
            
            }
        
        }

        System.out.println("Edges added due to high loss: " + edgesAdded);
        
        writer.println("Number of edges added by weight: " + edgesAdded);
    
    }

    public static void addEdgesByAccuracy(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next;
    
        int edgesAdded = 0;
    
        double newEdgeWeight = 0.005;
    
        double currentAccuracy = fnTrain.lastComputedAccuracy;
    
        double accuracyThreshold = 70.0;

        if (currentAccuracy < accuracyThreshold) {
        
            while (x != null && x.next != null) {
            
                node n = x.firstnode;
            
                while (n != null) {
                
                    edge newEdge = new edge();
                
                    newEdge.weight = newEdgeWeight;

                    if (x.next.firstnode != null) {
                    
                        newEdge.target = x.next.firstnode;
                    
                        newEdge.next = n.firstedge;
                    
                        n.firstedge = newEdge;
                    
                        edgesAdded++;
                    
                    }
                
                    n = n.next;
                
                }
            
                x = x.next;
            
            }
        
        }

        System.out.println("Edges added due to low accuracy: " + edgesAdded);  // Corrected the print message
        
        writer.println("Edges added due to low accuracy: " + edgesAdded);
    
    }



    public static void addNodesByLoss(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next;
    
        int nodesAdded = 0;

        double currentLoss = fnTrain.lastComputedLoss;
    
        double lossThreshold = 3.0;
    
        if (currentLoss > lossThreshold) {
        
            while (x != null && x.next != null) {
            
                node newNode = new node();
            
                newNode.firstedge = null;
            
                newNode.next = x.firstnode;
            
                x.firstnode = newNode;
            
                nodesAdded++;
            
                x = x.next;
            
            }
        
        }

        System.out.println("Nodes added due to low accuracy: " + nodesAdded);
        
        writer.println("Nodes added due to low accuracy: " + nodesAdded);
    
    }

    public static void addNodesByAccuracy(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next;
    
        int nodesAdded = 0;

        double currentAccuracy = fnTrain.lastComputedAccuracy;
     
        double accuracyThreshold = 70.0;

        if (currentAccuracy < accuracyThreshold) {
        
            while (x != null && x.next != null) {
            
                node newNode = new node();
            
                newNode.firstedge = null;
            
                newNode.next = x.firstnode;
            
                x.firstnode = newNode;
            
                nodesAdded++;
            
                x = x.next;
            
            }
        
        }

        System.out.println("Nodes added due to low accuracy: " + nodesAdded);
        
        writer.println("Nodes added due to low accuracy: " + nodesAdded);
    
    }







        

    public static void reinforceEdges(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode;
    
        int edgesReinforced = 0;
    
        double reinforcementWeight = 0.002;

        while (x != null) {
        
            node n = x.firstnode;
        
            while (n != null) {
            
                edge e = n.firstedge;
            
                while (e != null) {
                
                    if (Math.abs(e.weight) < reinforcementWeight) {
                    
                        e.weight += reinforcementWeight; // Strengthen weak edges
                    
                        edgesReinforced++;
                    
                    }
                
                    e = e.next;
                
                }
            
                n = n.next;
            
            }
        
            x = x.next;
        
        }

        System.out.println("Number of edges reinforced: " + edgesReinforced);
        
        writer.println("Number of edges reinforced: " + edgesReinforced);
    
    }



    public static void addEdgesToIsolatedNodes(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next; 
    
        int edgesAdded = 0;

        while (x != null && x.next != null) {
        
            node n = x.firstnode;
        
            while (n != null) {
            
                if (n.firstedge == null) { 
                
                    edge newEdge = new edge();
                
                    newEdge.weight = 0.002;

                    // Connect to a meaningful node in the next layer
                    if (x.next.firstnode != null) {
                    
                        newEdge.target = x.next.firstnode;
                    
                        newEdge.next = n.firstedge;
                    
                        n.firstedge = newEdge;
                    
                        edgesAdded++;
                    
                    }
                
                }
            
                n = n.next;
            
            }
        
            x = x.next;
        
        }

        System.out.println("Number of edges added to isolated nodes: " + edgesAdded);
        
        writer.println("Number of edges added to isolated nodes: " + edgesAdded);
    
    }


    
    public static void addNodes(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next;
    
        node newNode;
    
        int nodesAdded = 0;

        while (x.next != null) {
        
            if (x.firstnode == null) {
                
                newNode = new node();
            
                newNode.firstedge = null; 
            
                newNode.next = x.firstnode;
            
                x.firstnode = newNode;
            
                nodesAdded++;
            
            }
        
            x = x.next;
        
        }

        System.out.println("Number of nodes added is: " + nodesAdded);
        
        writer.println("Number of nodes added is: " + nodesAdded);
    
    }



    public static void populateEmptyLayers(network Net, PrintWriter writer) {
    
        layernode x = Net.inputlayernode.next;
    
        int nodesAdded = 0;

        while (x != null && x.next != null) {
        
            if (x.firstnode == null) {
           
                node newNode = new node();
            
                newNode.firstedge = null; // No edges yet
            
                newNode.next = x.firstnode;
            
                x.firstnode = newNode;
            
                nodesAdded++;
            
            }
        
            x = x.next;
        
        }

        System.out.println("Number of nodes added to empty layers: " + nodesAdded);
        
        writer.println("Number of nodes added to empty layers: " + nodesAdded);
    
        }


    }