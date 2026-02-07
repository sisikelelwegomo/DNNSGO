package dnnsg;

import java.io.*;
import java.util.*;

/**
 * REAL PARKINSON'S DATASET LOADER
 * 
 * Dataset: 196 samples, 22 input features, 1 output (diagnosis: 0 or 1)
 * Features: feature_0 through feature_21 (already normalized 0-1)
 * Output: diagnosis (0 = no Parkinson's, 1 = Parkinson's)
 * 
 * This loader:
 * 1. Reads actual CSV file
 * 2. Normalizes features (z-score normalization)
 * 3. Converts binary target to one-hot encoding if needed
 * 4. Splits data into train (70%) and validation (30%)
 * 5. Handles stratified splitting to maintain class distribution
 */
public class ParkinsonsDataLoader {
    
    // Dataset arrays
    public double[][] trainInputs;
    public double[][] trainTargets;
    public double[][] valInputs;
    public double[][] valTargets;
    
    // Statistics for normalization
    private double[] featureMeans;
    private double[] featureStdDev;
    
    // Dataset info
    private int numSamples;
    private int numFeatures;
    private int numOutputs;
    
    /**
     * Load the Parkinson's dataset from CSV file
     * @param csvPath - Full path to parkinsons.csv
     * @param inputSize - Number of input features (should be 22)
     * @param outputSize - Number of output classes (should be 1 for binary, or 2 for one-hot)
     */
    public void load(String csvPath, int inputSize, int outputSize) throws IOException {
        System.out.println("\n" + "â”€".repeat(80));
        System.out.println("LOADING PARKINSON'S DATASET");
        System.out.println("â”€".repeat(80));
        
        this.numFeatures = inputSize;
        this.numOutputs = outputSize;
        
        // Step 1: Read raw data from CSV
        System.out.println("Step 1: Reading CSV file: " + csvPath);
        List<double[]> rawFeatures = new ArrayList<>();
        List<Integer> rawLabels = new ArrayList<>();
        
        readCSV(csvPath, rawFeatures, rawLabels);
        
        this.numSamples = rawFeatures.size();
        System.out.println("  âœ“ Loaded " + numSamples + " samples with " + numFeatures + " features");
        
        // Step 2: Calculate normalization statistics
        System.out.println("Step 2: Computing normalization statistics (z-score)");
        calculateNormalizationStats(rawFeatures);
        System.out.println("  âœ“ Computed means and standard deviations");
        
        // Step 3: Normalize features
        System.out.println("Step 3: Normalizing features");
        double[][] normalizedFeatures = normalizeFeatures(rawFeatures);
        System.out.println("  âœ“ Features normalized");
        
        // Step 4: Convert labels to targets
        System.out.println("Step 4: Converting labels to target format");
        double[][] targets = convertLabelsToTargets(rawLabels, outputSize);
        System.out.println("  âœ“ Labels converted to " + outputSize + "-dimensional targets");
        
        // Step 5: Stratified train/validation split
        System.out.println("Step 5: Splitting data (70% train, 30% validation - stratified)");
        splitDataStratified(normalizedFeatures, targets, rawLabels, 0.7);
        System.out.println("  âœ“ Train: " + trainInputs.length + " samples");
        System.out.println("  âœ“ Validation: " + valInputs.length + " samples");
        
        // Step 6: Print class distribution
        System.out.println("\nStep 6: Class Distribution");
        printClassDistribution(rawLabels);
        
        System.out.println("â”€".repeat(80) + "\n");
    }
    
    /**
     * Read CSV file and extract features and labels
     */
    private void readCSV(String csvPath, List<double[]> features, List<Integer> labels) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String headerLine = br.readLine(); // Skip header
            String line;
            int lineNumber = 1;
            
            while ((line = br.readLine()) != null) {
                lineNumber++;
                line = line.trim();
                if (line.isEmpty()) continue;
                
                try {
                    String[] parts = line.split(",");
                    
                    // Extract features (first numFeatures columns)
                    double[] featureVector = new double[numFeatures];
                    for (int i = 0; i < numFeatures; i++) {
                        featureVector[i] = Double.parseDouble(parts[i]);
                    }
                    features.add(featureVector);
                    
                    // Extract label (last column - diagnosis)
                    int label = (int) Double.parseDouble(parts[numFeatures]);
                    labels.add(label);
                    
                } catch (NumberFormatException e) {
                    System.err.println("Warning: Could not parse line " + lineNumber + ": " + line);
                }
            }
        }
    }
    
    /**
     * Calculate mean and standard deviation for each feature (for z-score normalization)
     */
    private void calculateNormalizationStats(List<double[]> features) {
        featureMeans = new double[numFeatures];
        featureStdDev = new double[numFeatures];
        
        // Calculate means
        for (double[] feature : features) {
            for (int i = 0; i < numFeatures; i++) {
                featureMeans[i] += feature[i];
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureMeans[i] /= features.size();
        }
        
        // Calculate standard deviations
        for (double[] feature : features) {
            for (int i = 0; i < numFeatures; i++) {
                double diff = feature[i] - featureMeans[i];
                featureStdDev[i] += diff * diff;
            }
        }
        for (int i = 0; i < numFeatures; i++) {
            featureStdDev[i] = Math.sqrt(featureStdDev[i] / features.size());
            // Avoid division by zero
            if (featureStdDev[i] == 0) {
                featureStdDev[i] = 1.0;
            }
        }
    }
    
    /**
     * Apply z-score normalization: (x - mean) / stddev
     */
    private double[][] normalizeFeatures(List<double[]> features) {
        double[][] normalized = new double[features.size()][numFeatures];
        
        for (int i = 0; i < features.size(); i++) {
            double[] original = features.get(i);
            for (int j = 0; j < numFeatures; j++) {
                normalized[i][j] = (original[j] - featureMeans[j]) / featureStdDev[j];
            }
        }
        
        return normalized;
    }
    
    /**
     * Convert integer labels to target format
     * If outputSize == 1: keep as [0.0] or [1.0]
     * If outputSize == 2: convert to one-hot [1.0, 0.0] or [0.0, 1.0]
     */
    private double[][] convertLabelsToTargets(List<Integer> labels, int outputSize) {
        double[][] targets = new double[labels.size()][outputSize];
        
        for (int i = 0; i < labels.size(); i++) {
            int label = labels.get(i);
            
            if (outputSize == 1) {
                // Single output: use the label directly
                targets[i][0] = (double) label;
            } else if (outputSize == 2) {
                // One-hot encoding: [0, 1] for class 1, [1, 0] for class 0
                if (label == 0) {
                    targets[i][0] = 1.0;
                    targets[i][1] = 0.0;
                } else {
                    targets[i][0] = 0.0;
                    targets[i][1] = 1.0;
                }
            }
        }
        
        return targets;
    }
    
    /**
     * Stratified train/validation split
     * Maintains class distribution in both sets
     */
    private void splitDataStratified(double[][] features, double[][] targets, 
                                     List<Integer> labels, double trainRatio) {
        // Separate indices by class
        List<Integer> class0Indices = new ArrayList<>();
        List<Integer> class1Indices = new ArrayList<>();
        
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i) == 0) {
                class0Indices.add(i);
            } else {
                class1Indices.add(i);
            }
        }
        
        // Shuffle within each class
        Collections.shuffle(class0Indices, new Random(42)); // Fixed seed for reproducibility
        Collections.shuffle(class1Indices, new Random(42));
        
        // Split each class
        int trainSize0 = (int) (class0Indices.size() * trainRatio);
        int trainSize1 = (int) (class1Indices.size() * trainRatio);
        
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> valIndices = new ArrayList<>();
        
        // Add training samples
        for (int i = 0; i < trainSize0; i++) {
            trainIndices.add(class0Indices.get(i));
        }
        for (int i = 0; i < trainSize1; i++) {
            trainIndices.add(class1Indices.get(i));
        }
        
        // Add validation samples
        for (int i = trainSize0; i < class0Indices.size(); i++) {
            valIndices.add(class0Indices.get(i));
        }
        for (int i = trainSize1; i < class1Indices.size(); i++) {
            valIndices.add(class1Indices.get(i));
        }
        
        // Create train and validation datasets
        trainInputs = new double[trainIndices.size()][numFeatures];
        trainTargets = new double[trainIndices.size()][numOutputs];
        valInputs = new double[valIndices.size()][numFeatures];
        valTargets = new double[valIndices.size()][numOutputs];
        
        // Fill train data
        for (int i = 0; i < trainIndices.size(); i++) {
            int idx = trainIndices.get(i);
            trainInputs[i] = features[idx];
            trainTargets[i] = targets[idx];
        }
        
        // Fill validation data
        for (int i = 0; i < valIndices.size(); i++) {
            int idx = valIndices.get(i);
            valInputs[i] = features[idx];
            valTargets[i] = targets[idx];
        }
    }
    
    /**
     * Print class distribution statistics
     */
    private void printClassDistribution(List<Integer> labels) {
        int count0 = 0, count1 = 0;
        
        for (int label : labels) {
            if (label == 0) count0++;
            else count1++;
        }
        
        System.out.println("  Class 0 (No Parkinson's): " + count0 + " samples (" + 
                          String.format("%.1f%%", 100.0 * count0 / labels.size()) + ")");
        System.out.println("  Class 1 (Parkinson's): " + count1 + " samples (" + 
                          String.format("%.1f%%", 100.0 * count1 / labels.size()) + ")");
    }
    
    /**
     * Get normalization statistics for reference
     */
    public void printNormalizationStats() {
        System.out.println("\nNormalization Statistics (z-score):");
        System.out.println("Feature means: " + Arrays.toString(featureMeans));
        System.out.println("Feature std devs: " + Arrays.toString(featureStdDev));
    }
    
    /**
     * Utility: Denormalize a feature value back to original scale
     */
    public double denormalize(double normalizedValue, int featureIndex) {
        return normalizedValue * featureStdDev[featureIndex] + featureMeans[featureIndex];
    }
}