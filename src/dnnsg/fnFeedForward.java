package dnnsg;

import java.io.PrintWriter;

/**
 * Feedforward class for neural network propagation with skip connections.
 * 
 * @author Clint van Alten
 * @author Sisikelelwe Gomo
 */
public class fnFeedForward {
    
    // Add this method to apply dropout before forward propagation
    public static void applyDropout(network Net, double dropoutRate, boolean isTraining, PrintWriter writer) {
        // Remove this line: dropoutRate = 0.2; // Don't overwrite the parameter!
        
        if (!isTraining || dropoutRate <= 0) {
            // Reset all dropout flags during inference or when dropout is disabled
            layernode layer = Net.inputlayernode;
            while (layer != null) {
                node n = layer.firstnode;
                while (n != null) {
                    n.isDroppedOut = false;
                    n = n.next;
                }
                layer = layer.next;
            }
            writer.println("Dropout Rate: 0.0 (disabled)");
            return;
        }
        
        int totalNodes = 0;
        int droppedNodes = 0;
        
        // Apply dropout randomly during training (to hidden layers only)
        layernode layer = Net.inputlayernode.next; // Skip input layer
        while (layer != null && layer != Net.outputlayernode) { // Apply to hidden layers only
            node n = layer.firstnode;
            while (n != null) {
                n.isDroppedOut = (Math.random() < dropoutRate);
                if (n.isDroppedOut) {
                    droppedNodes++;
                }
                totalNodes++;
                n = n.next;
            }
            layer = layer.next;
        }
        
        // Record actual dropout statistics
        double actualRate = totalNodes > 0 ? (double) droppedNodes / totalNodes : 0.0;
        writer.println("Dropout - Target: " + dropoutRate + ", Actual: " + actualRate + 
                      ", Dropped: " + droppedNodes + "/" + totalNodes + " nodes");
    }
    
    
    // Standard FeedForward without skip connections
public static void feedForward(network Net, double[] A) {
    // Apply dropout first (you'll need to pass dropoutRate and isTraining)
    // applyDropout(Net, dropoutRate, true); // Uncomment when you have the parameters
    
    // Process the input layer
    layernode x = Net.inputlayernode;
    node n = x.firstnode;
    int i = 0;

    while (n != null) {
        // Respect dropout - if node is dropped, set activation to 0
        if (n.isDroppedOut) {
            n.actvalue = 0.0;
        } else {
            n.actvalue = A[i]; // Assign input values
        }
        fireNode(n);       // Propagate to connected nodes
        i++;
        n = n.next;
    }

    // Process hidden and output layers
    x = x.next;
    while (x != null) {
        n = x.firstnode;

        while (n != null) {
            // Respect dropout - if node is dropped, skip computation
            if (n.isDroppedOut) {
                n.actvalue = 0.0;
            } else {
                // Compute raw activation (weighted sum + bias)
                n.actvalue = n.sum + n.bias;

                // Apply activation for hidden layers only
                switch (n.type) {
                    case 0 -> {
                    }
                    case 1 -> {
                        // ReLU
                        if (n.actvalue < 0.0) n.actvalue = 0.0;
                    }
                    case 2 -> // Sigmoid
                        n.actvalue = 1.0 / (1.0 + Math.exp(-n.actvalue));
                    case 3 -> // Tanh
                        n.actvalue = Math.tanh(n.actvalue);
                }
                // Linear - do nothing
                            }
            
            fireNode(n); // Propagate activations to connected nodes
            n.sum = 0.0; // Reset sum after firing node
            n = n.next;
        }
        x = x.next;
    }
    
}

    
    
    // Standard FeedForward without skip connections
    public static void feedForward(network Net, double[] A, double dropoutRate, boolean isTraining, PrintWriter writer) {
        // Apply dropout first
        applyDropout(Net, dropoutRate, isTraining, writer);
        
        // Process the input layer
        layernode x = Net.inputlayernode;
        node n = x.firstnode;
        int i = 0;

        while (n != null) {
            // Respect dropout - if node is dropped, set activation to 0
            if (n.isDroppedOut) {
                n.actvalue = 0.0;
            } else {
                n.actvalue = A[i]; // Assign input values
            }
            fireNode(n);       // Propagate to connected nodes
            i++;
            n = n.next;
        }

        // Process hidden and output layers
        x = x.next;
        while (x != null) {
            n = x.firstnode;

            while (n != null) {
                // Respect dropout - if node is dropped, skip computation
                if (n.isDroppedOut) {
                    n.actvalue = 0.0;
                } else {
                    // Compute raw activation (weighted sum + bias)
                    n.actvalue = n.sum + n.bias;

                    // Apply activation for hidden layers only
                    switch (n.type) {
                        case 0 -> {
                        }
                        case 1 -> {
                            // ReLU
                            if (n.actvalue < 0.0) n.actvalue = 0.0;
                        }
                        case 2 -> // Sigmoid
                            n.actvalue = 1.0 / (1.0 + Math.exp(-n.actvalue));
                        case 3 -> // Tanh
                            n.actvalue = Math.tanh(n.actvalue);
                    }
                    // Linear - do nothing
                }
                
                fireNode(n); // Propagate activations to connected nodes
                n.sum = 0.0; // Reset sum after firing node
                n = n.next;
            }
            x = x.next;
        }
    }

    // Method to propagate node activation to connected nodes
    public static void fireNode(node n) {
        for (edge e = n.firstedge; e != null; e = e.next) {
            e.target.sum += n.actvalue * e.weight;
        }
    }

    // Helper method to get network outputs
    public static double[] getOutputs(network Net, int outputLength) {
        double[] T = new double[outputLength];
        layernode x = Net.outputlayernode;
        node n = x.firstnode;
        int i = 0;
        while (n != null) {
            T[i] = n.actvalue;
            i++;
            n = n.next;
        }
        return T;
    }
}