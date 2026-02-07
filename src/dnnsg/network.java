package dnnsg;

/**
 *
 * @author Clint van Alten
 */
public class network {
    
    layernode inputlayernode;
    
    layernode outputlayernode;
    
        public int getNumberOfLayers() {
            
        int layerCount = 0;
        
        layernode currentLayer = inputlayernode; // Start from the input layer

        // Traverse through the linked list of layers and count them
        
        while (currentLayer != null) {
            
            layerCount++;
            
            currentLayer = currentLayer.next; // Move to the next layer
            
        }

        return layerCount; // Return the layer count
    }
        
        
        public static void assignLayerIndices(network Net) {
    layernode current = Net.inputlayernode;
    int index = 0;

    while (current != null) {
        current.setIndex(index);
        current = current.next;
        index++;
    }
}


}