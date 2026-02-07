/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package dnnsg;

/**
 *
 * @author jimmy
 */
import java.util.*;

public class SMOTE {
    // K-nearest neighbors implementation for SMOTE
    private static List<double[]> findKNearestNeighbors(double[][] data, double[] sample, int k) {
        List<double[]> neighbors = new ArrayList<>();
        PriorityQueue<double[]> pq = new PriorityQueue<>(
            (a, b) -> Double.compare(distance(sample, b), distance(sample, a))
        );
        
        for (double[] point : data) {
            pq.offer(point);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        neighbors.addAll(pq);
        return neighbors;
    }
    
    private static double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    // Generate synthetic samples
    public static double[][] generateSyntheticSamples(double[][] minoritySamples, int k, int numToGenerate) {
        List<double[]> syntheticSamples = new ArrayList<>();
        Random rand = new Random();
        
        for (int i = 0; i < numToGenerate; i++) {
            // Randomly select a minority sample
            double[] sample = minoritySamples[rand.nextInt(minoritySamples.length)];
            
            // Find its k nearest neighbors
            List<double[]> neighbors = findKNearestNeighbors(minoritySamples, sample, k);
            double[] neighbor = neighbors.get(rand.nextInt(neighbors.size()));
            
            // Generate synthetic sample
            double[] synthetic = new double[sample.length];
            double gap = rand.nextDouble(); // Random number between 0 and 1
            for (int j = 0; j < synthetic.length; j++) {
                synthetic[j] = sample[j] + gap * (neighbor[j] - sample[j]);
            }
            
            syntheticSamples.add(synthetic);
        }
        
        return syntheticSamples.toArray(new double[0][]);
    }
}
