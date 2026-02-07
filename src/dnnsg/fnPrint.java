package dnnsg;

/**
 *
 * @author Clint van Alten
 */
public class fnPrint {
    
    public static void printOutputs(network Net) {
    
        node n = Net.outputlayernode.firstnode;
        
        System.out.println();
        
        while (n != null) {
    
            System.out.print(n.actvalue+", ");
        
            n = n.next;
        }
    }


    public static void printNetwork(network Net) {
    
        countLayers(Net);
        
        countNodes(Net);
        
        assignIndices(Net);
        
        printNodesAndEdges(Net);
    }
    

    public static void countLayers(network Net) {

        layernode x = Net.inputlayernode;

        int layercount = 0;
        
        while (x != null) {
        
            layercount++;

            x = x.next;
        }

        System.out.println();

        System.out.println("Number of layers: "+ layercount);
}    



    public static void countNodes(network Net) {

        layernode x = Net.inputlayernode;
    
        node n;
        
        int layer = 0;
        
        int nodecount;
        
        System.out.println();
        
        while (x != null) {
            
            n = x.firstnode;
        
            nodecount = 0;
            
            while (n != null) {
                                
                nodecount++;
        
                n = n.next;
            }
            
            System.out.println("Nodes in layer "+layer+": "+nodecount);

            x = x.next;
            
            layer++; 
        }
    }  



    public static void assignIndices(network Net) {

        layernode x = Net.inputlayernode;
    
        node n;
    
        int lindex = 0;

        int nindex = 0;
    
        while (x != null) {
    
            n = x.firstnode;

            nindex = 0;
            
            while (n != null) {
            
                n.layerindex = lindex;

                n.nodeindex = nindex;
            
                nindex++;

                n = n.next;
            }
        
            lindex++;

            x = x.next;
    }
}    
    
    
    public static void printNodesAndEdges(network Net) {
    
        layernode x = Net.inputlayernode;
        
        int edgecount;
        
        node n;

        edge e;
        
        System.out.println();
        
        while (x != null) {
            
            n = x.firstnode;
            
            while (n != null) {
                
                System.out.println("node ("+n.layerindex+","+n.nodeindex+"): "
                        + "type = "+n.type +", bias = "+n.bias+", actvalue = "+n.actvalue);
                
                System.out.println("Edges:");
                
                e = n.firstedge;

                edgecount = 0;
                
                while (e != null) {
                
                    System.out.println(edgecount+": weight = "+e.weight+
                        ", target = ("+e.target.layerindex+","+e.target.nodeindex+")");

                    e = e.next;

                    edgecount++;
                }

                n = n.next;
            }
 
            x = x.next; 
        }
    }

}