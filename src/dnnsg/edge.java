package dnnsg;

/**
 *
 * @author Clint van Alten
 */
public class edge {
    public node target;  // The target node this edge connects to
    public edge next;    // Pointer to the next edge in the list
    public double weight; // The weight of the edge
    public boolean isSkip; // Flag to indicate if it's a skip connection
    public node source;  // The source node this edge originates from
    // Add these fields to the edge class
double m;  // First moment estimate (mean)
double v;  // Second moment estimate (uncentered variance)
    // Constructor for regular connections
    public edge(node source, node target) {
        this.source = source;
        this.target = target;
        this.weight = (4*Math.random() - 2);
 // Random initial weight
        this.isSkip = false; // Default: not a skip connection
    }

    // Constructor for explicitly setting skip flag
    public edge(node source, node target, boolean isSkip) {
        this.source = source;
        this.target = target;
        this.weight = (4*Math.random() - 2);
        this.isSkip = isSkip;
    }
    
    
        
    public edge() {
    
        this.weight = (4*Math.random() - 2);

    }
}