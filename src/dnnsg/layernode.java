package dnnsg;

/**
 *
 * @author Clint van Alten
 */
public class layernode {
    
    layernode next;
    
    layernode prev;

    node firstnode;   // first node in the list of nodes in this layer
    
     int index;
    
    
     public int getNodeCount() {
    int count = 0;
    node currentNode = this.firstnode;
    while (currentNode != null) {
        count++;
        currentNode = currentNode.next;
    }
    return count;
}
     
     // Getter for the layer index
// Returns the index of the layer. The index represents the position of the layer in the network.
public int getIndex() {
    return index;
}

// Setter for the layer index
// Updates the index of the layer. This can be used if the layer's position in the network changes.
public void setIndex(int index) {
    this.index = index;         // Set the layer's index to the given value
}

public int getEdgeCount() {
    if (this.next == null) {
        return 0; // Output layer has no outgoing edges
    }
    return this.getNodeCount() * this.next.getNodeCount();
}

}