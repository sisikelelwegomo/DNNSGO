package dnnsg;

public class RMSpropOptimizer {
    private final double alpha;  // Learning rate
    private final double gamma; // Decay rate for moving average of squared gradients
    private final double epsilon; // Small number to prevent division by zero
    
    public RMSpropOptimizer(double alpha, double gamma, double epsilon) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
    }
    
    public void updateWeights(network net) {
        layernode layer = net.inputlayernode;
        
        while (layer != null) {
            node n = layer.firstnode;
            while (n != null) {
                // Update node bias using RMSProp
                updateParameter(n, true);
                
                // Update outgoing edge weights using RMSProp
                edge e = n.firstedge;
                while (e != null) {
                    updateParameter(e);
                    e = e.next;
                }
                
                n = n.next;
            }
            layer = layer.next;
        }
    }
    
    private void updateParameter(edge e) {
        // Calculate squared gradient
        double gradient = e.target.delta * e.source.actvalue;
        
        // Update moving average of squared gradients
        e.v = gamma * e.v + (1 - gamma) * Math.pow(gradient, 2);
        
        // Update weight using RMSProp formula
        e.weight -= alpha * gradient / (Math.sqrt(e.v) + epsilon);
    }
    
    private void updateParameter(node n, boolean isBias) {
        if (isBias) {
            // Calculate squared gradient for bias
            double gradient = n.delta;
            
            // Update moving average of squared gradients for bias
            n.v_bias = gamma * n.v_bias + (1 - gamma) * Math.pow(gradient, 2);
            
            // Update bias using RMSProp formula
            n.bias -= alpha * gradient / (Math.sqrt(n.v_bias) + epsilon);
        }
    }
}