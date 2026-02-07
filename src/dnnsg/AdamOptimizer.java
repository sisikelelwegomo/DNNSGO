package dnnsg;

public class AdamOptimizer {
    private double alpha;  // Learning rate
    private double beta1; // Exponential decay rate for first moment estimates
    private double beta2; // Exponential decay rate for second moment estimates
    private double epsilon; // Small number to prevent division by zero
    private int t; // Time step counter
    
    public AdamOptimizer(double alpha, double beta1, double beta2, double epsilon) {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
    }
    
    public void updateWeights(network net) {
        t++;
        layernode layer = net.inputlayernode;
        
        while (layer != null) {
            node n = layer.firstnode;
            while (n != null) {
                // Update node bias
                updateParameter(n, true);
                
                // Update outgoing edge weights
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
        // Update first moment estimate
        e.m = beta1 * e.m + (1 - beta1) * e.target.delta * e.source.actvalue;
        
        // Update second moment estimate
        e.v = beta2 * e.v + (1 - beta2) * Math.pow(e.target.delta * e.source.actvalue, 2);
        
        // Compute bias-corrected estimates
        double m_hat = e.m / (1 - Math.pow(beta1, t));
        double v_hat = e.v / (1 - Math.pow(beta2, t));
        
        // Update weight
        e.weight -= alpha * m_hat / (Math.sqrt(v_hat) + epsilon);
    }
    
    private void updateParameter(node n, boolean isBias) {
        if (isBias) {
            // Update first moment estimate for bias
            n.m_bias = beta1 * n.m_bias + (1 - beta1) * n.delta;
            
            // Update second moment estimate for bias
            n.v_bias = beta2 * n.v_bias + (1 - beta2) * Math.pow(n.delta, 2);
            
            // Compute bias-corrected estimates
            double m_hat = n.m_bias / (1 - Math.pow(beta1, t));
            double v_hat = n.v_bias / (1 - Math.pow(beta2, t));
            
            // Update bias
            n.bias -= alpha * m_hat / (Math.sqrt(v_hat) + epsilon);
        }
    }
}