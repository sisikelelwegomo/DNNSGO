package dnnsg;

/**
 *
 * @author Clint van Alten
 * @author Sisikelelwe Gomo
 */

public class node {
 
    // Add these fields to the node class
    
    double m;  // First moment estimate (mean)
    
    double v;  // Second moment estimate (uncentered variance)
    
    double m_bias;  // First moment estimate for bias
    
    double v_bias;  // Second moment estimate for bias

    double bias;
    
    double sum;
    
    double actvalue;
    
    double delta;
    
    int type;  // defines the activation function
    
    int layerindex;  // used when printing or saving the network
    
    int nodeindex;   // used when printing or saving the network
    
    boolean markedfordeletion;   // used when deleting this node
    
    boolean markedforddition;   // used when adding this node
    
    edge firstedge;  //  first edge in list of edges with this node as head
    
    node next;   // next node in this layer
    
    layernode layer;
    
    int activation;
    
    boolean isDroppedOut;
    
    public double v_gamma;
    
    public double v_beta;
    
    public double beta;
    
    public double gamma;
    
    public double normalizedValue;

    public node(int t) {
    
        if (t == -1) this.type = (int)(Math.random() * 4);
        
        else  this.type = t;
        
        this.bias = (2*Math.random() - 1);
        
        this.sum = 0.0;
        
        this.actvalue = 0.0;
        
        this.delta = 0.0; 
        
        this.markedfordeletion = false;
        
        this.markedforddition = false;
        
        this.layerindex = 0;
        
        this.nodeindex = 0;
        
        this.m = 0.0;
        
        this.v = 0.0;
        
        this.m_bias = 0.0;
        
        this.v_bias = 0.0;
        
        this.isDroppedOut = false;
        
        this.gamma = 1.0;      // Scale parameter
        this.beta = 0.0;       // Shift parameter
     
        // RMSprop accumulator
        this.v_bias = 0.0;      
        this.v_gamma = 0.0;     
        this.v_beta = 0.0;  
        
        
        
        
    }
    
        public node() {
        
        this.bias = (2*Math.random() - 1);
        
        this.sum = 0.0;
        
        this.actvalue = 0.0;
        
        this.delta = 0.0; 
        
        this.markedfordeletion = false;
        
        this.markedforddition = false;
        
        this.layerindex = 0;
        
        this.nodeindex = 0;
        
        this.m = 0.0;
        
        this.v = 0.0;
        
        this.m_bias = 0.0;
        
        this.v_bias = 0.0;
        
        this.isDroppedOut = false;
    }
    

}