package dnnsg;

public class fnRemove {
   public static void networkCleanup(network Net) {
      deleteEdgesByWeight(Net);
      markNodesForDeletion(Net);
      deleteNodes(Net);
      deleteLayernodes(Net);
   }

   public static void deleteEdgesByWeight(network Net) {
      layernode x = Net.outputlayernode.prev;
      double minweight = 0.001D;

      int edgesremoved;
      for(edgesremoved = 0; x != null; x = x.prev) {
         for(node n = x.firstnode; n != null; n = n.next) {
            edge e;
            for(e = n.firstedge; e != null && Math.abs(e.weight) < minweight; ++edgesremoved) {
               e = e.next;
            }

            n.firstedge = e;
            if (e == null) {
               n.markedfordeletion = true;
            }

            while(e != null) {
               edge d;
               for(d = e.next; d != null && Math.abs(e.weight) < minweight; ++edgesremoved) {
                  d = d.next;
               }

               e.next = d;
               e = d;
            }
         }
      }

      System.out.println("Number of edges removed by weight is: " + edgesremoved);
   }

   public static void deleteEdgesByTarget(network Net) {
      layernode x = Net.outputlayernode.prev;

      int edgesremoved;
      for(edgesremoved = 0; x != null; x = x.prev) {
         for(node n = x.firstnode; n != null; n = n.next) {
            edge e;
            for(e = n.firstedge; e != null && e.target.markedfordeletion; ++edgesremoved) {
               e = e.next;
            }

            edge d;
            for(n.firstedge = e; e != null; e = d) {
               for(d = e.next; d != null && d.target.markedfordeletion; ++edgesremoved) {
                  d = d.next;
               }

               e.next = d;
            }
         }
      }

      System.out.println("Number of edges removed by target is: " + edgesremoved);
   }

   public static void markNodesForDeletion(network Net) {
      layernode x = Net.outputlayernode.prev;

      int nodesmarked;
      for(nodesmarked = 0; x.prev != null; x = x.prev) {
         for(node n = x.firstnode; n != null; n = n.next) {
            if (n.firstedge == null) {
               n.markedfordeletion = true;
               ++nodesmarked;
            }
         }
      }

      System.out.println("Number of nodes marked for deletion is: " + nodesmarked);
   }

   public static void deleteNodes(network Net) {
      deleteEdgesByTarget(Net);
      layernode x = Net.inputlayernode.next;

      int nodesremoved;
      for(nodesremoved = 0; x.next != null; x = x.next) {
         node n;
         for(n = x.firstnode; n != null && n.markedfordeletion; ++nodesremoved) {
            n = n.next;
         }

         node m;
         for(x.firstnode = n; n != null; n = m) {
            for(m = n.next; m != null && m.markedfordeletion; ++nodesremoved) {
               m = m.next;
            }

            n.next = m;
         }
      }

      System.out.println("Number of nodes removed is: " + nodesremoved);
   }

   public static void deleteLayernodes(network Net) {
      layernode x = Net.inputlayernode.next;

      int layernodesremoved;
      for(layernodesremoved = 0; x.next != null; x = x.next) {
         if (x.firstnode == null) {
            x.prev.next = x.next;
            x.next.prev = x.prev;
            ++layernodesremoved;
         }
      }

      System.out.println("The number of layernodes removed is " + layernodesremoved);
   }
}