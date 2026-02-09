package dnnsgo;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DNNSGO {

    // Optimizer configurations to test
    static OptimizerConfig[] OPTIMIZER_CONFIGS = {
        // ADAM (4 configurations)
        new OptimizerConfig("HP1_Adam_Baseline", "ADAM", 0.001, 0.0005, 0.99, 0.999, 0.99),
        new OptimizerConfig("HP2_Adam_HighLR", "ADAM", 0.01, 0.0015, 0.99, 0.999, 0.99),
        new OptimizerConfig("HP3_Adam_LowLR", "ADAM", 0.0005, 0.0002, 0.99, 0.999, 0.99),
        new OptimizerConfig("HP4_Adam_HighL2", "ADAM", 0.005, 0.0010, 0.99, 0.999, 0.99),
        
        // RMSPROP (4 configurations)
        new OptimizerConfig("HP5_RMSprop_Baseline", "RMSPROP", 0.001, 0.0005, 0.99, 0.999, 0.99),
        new OptimizerConfig("HP6_RMSprop_HighLR", "RMSPROP", 0.01, 0.0015, 0.99, 0.999, 0.99),
        new OptimizerConfig("HP7_RMSprop_LowDecay", "RMSPROP", 0.0008, 0.0003, 0.99, 0.999, 0.95),
        new OptimizerConfig("HP8_RMSprop_HighDecay", "RMSPROP", 0.003, 0.0009, 0.99, 0.999, 0.999),
        
        // L2/SGD (4 configurations)
        new OptimizerConfig("HP9_L2_Baseline", "L2", 0.001, 0.0005, 0.0, 0.0, 0.0),
        new OptimizerConfig("HP10_L2_HighLR", "L2", 0.01, 0.0015, 0.0, 0.0, 0.0),
        new OptimizerConfig("HP11_L2_LowLR", "L2", 0.0005, 0.0002, 0.0, 0.0, 0.0),
        new OptimizerConfig("HP12_L2_HighL2", "L2", 0.005, 0.0010, 0.0, 0.0, 0.0)
    };
    
    // Current optimizer being tested
    static OptimizerConfig CURRENT_OPTIMIZER = null;
    
    // Store baseline parameter counts for each input/output class
    static Map<String, Integer> CLASS_BASELINE_PARAMS = new HashMap<>();
    static Map<String, String> BASELINE_STUDENT_CONFIG = new HashMap<>();
    static final double MAX_PARAM_DEVIATION = 0.05;

    // Network configurations
    static int[] inputsize = {21, 15, 9};
    static int[] outputsize = {10, 8, 4};
    static int NUM_REPLICATIONS = 3;

    // Teacher configurations (~15K params each)
    static int[][] teacherConfigs = {
        {21, 270, 10}, {21, 130, 85, 10}, {21, 80, 80, 80, 10},
        {21, 270, 4}, {21, 130, 85, 4}, {21, 80, 80, 80, 4},
        {15, 320, 10}, {15, 150, 100, 10}, {15, 90, 90, 90, 10},
        {15, 320, 4}, {15, 150, 100, 4}, {15, 90, 90, 90, 4},
        {9, 410, 10}, {9, 200, 130, 10}, {9, 120, 120, 120, 10},
        {9, 410, 4}, {9, 200, 130, 4}, {9, 120, 120, 120, 4},
        {9, 410, 8}, {9, 200, 130, 8}, {9, 120, 120, 120, 8},
        {15, 320, 8}, {15, 150, 100, 8}, {15, 90, 90, 90, 8} 
    };

    // Student configurations (all will be matched to baseline)
    static int[][] studentConfigs = {
        // 21 inputs, 10 outputs
        {21, 16, 16, 10},           // 3 layers
        {21, 12, 12, 12, 10},       // 4 layers
        {21, 10, 10, 10, 10, 10},   // 5 layers
        {21, 8, 8, 8, 8, 8, 10},    // 6 layers
        {21, 6, 6, 6, 6, 6, 6, 10}, // 7 layers
        {21, 5, 5, 5, 5, 5, 5, 5, 10}, // 8 layers
        // Add configurations for other input/output combinations as needed
    };
    
    static int numdatapoints = 10000;
    static int numValdatapoints = 1000;
    static int numepochs = 200;

    private static void ensureDirectoryExists(String filePath) {
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
    }

    // Optimizer configuration class
    static class OptimizerConfig {
        String name;
        String type; // "ADAM", "RMSPROP", "SGD", "L2"
        double learningRate;
        double l2Lambda;
        double beta1;
        double beta2;
        double decayRate;
        
        OptimizerConfig(String name, String type, double lr, double l2, double b1, double b2, double decay) {
            this.name = name;
            this.type = type;
            this.learningRate = lr;
            this.l2Lambda = l2;
            this.beta1 = b1;
            this.beta2 = b2;
            this.decayRate = decay;
        }
        
        @Override
        public String toString() {
            return String.format("%s: %s (lr=%.4f, l2=%.4f, b1=%.3f, b2=%.3f, decay=%.3f)", 
                name, type, learningRate, l2Lambda, beta1, beta2, decayRate);
        }
    }

    private static String getClassKey(int input, int output) {
        return input + "_" + output;
    }

    private static String configToString(int[] config) {
        return java.util.Arrays.toString(config);
    }

    // Calculate baseline parameters for each class (with 0% skip)
    public static void calculateBaselineParameters() {
        System.out.println("\n" + "═".repeat(80));
        System.out.println("PHASE 1: CALCULATING STUDENT BASELINES BY CLASS (0% SKIP)");
        System.out.println("═".repeat(80) + "\n");
        
        String baselineLogPath = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\OptimizerExperiments\\baseline_params.txt";
        ensureDirectoryExists(baselineLogPath);
        
        try (PrintWriter baselineWriter = new PrintWriter(new FileWriter(baselineLogPath))) {
            baselineWriter.println("STUDENT BASELINE PARAMETERS BY CLASS (0% Skip Connections)");
            baselineWriter.println("First student in each (input, output) class serves as baseline");
            baselineWriter.println();
            
            for (int input : inputsize) {
                for (int output : outputsize) {
                    List<int[]> matchingStudents = getMatchingStudents(input, output);
                    
                    if (matchingStudents.isEmpty()) continue;
                    
                    String classKey = getClassKey(input, output);
                    
                    System.out.println("\nInput=" + input + ", Output=" + output + ":");
                    baselineWriter.println("Input=" + input + ", Output=" + output + ":");
                    
                    // FIRST student in this class is the baseline
                    int[] baselineConfig = matchingStudents.get(0);
                    int baselineParams = createAndCountParams(baselineConfig, 0.0);
                    
                    CLASS_BASELINE_PARAMS.put(classKey, baselineParams);
                    BASELINE_STUDENT_CONFIG.put(classKey, configToString(baselineConfig));
                    
                    String archStr = architectureToString(baselineConfig);
                    String depthStr = getStudentDepth(baselineConfig);
                    
                    System.out.println("  ✓ BASELINE: " + archStr + " (" + depthStr + "): " + 
                                     baselineParams + " params");
                    
                    baselineWriter.println("  ✓ BASELINE: " + archStr + " (" + depthStr + "): " + 
                                         baselineParams + " params");
                    baselineWriter.println("    (All other students in this class will match this count ±5%)");
                    baselineWriter.println();
                    
                    // Show other students
                    for (int i = 1; i < matchingStudents.size(); i++) {
                        int[] otherConfig = matchingStudents.get(i);
                        String otherArch = architectureToString(otherConfig);
                        String otherDepth = getStudentDepth(otherConfig);
                        
                        System.out.println("  • Other: " + otherArch + " (" + otherDepth + ") will be adjusted to " + 
                                         baselineParams + " params");
                        baselineWriter.println("  • Other: " + otherArch + " (" + otherDepth + ") will be adjusted");
                    }
                    baselineWriter.println();
                }
            }
            
            baselineWriter.println("Total Classes: " + CLASS_BASELINE_PARAMS.size());
            
        } catch (IOException e) {
            System.out.println("Error writing baseline parameters file: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n" + "═".repeat(80));
        System.out.println(" STUDENT BASELINES CALCULATED: " + CLASS_BASELINE_PARAMS.size() + " classes");
        System.out.println("═".repeat(80) + "\n");
    }

    public static void main(String[] args) {
        try {
            System.out.println("\n" + "═".repeat(80));
            System.out.println("OPTIMIZER COMPARISON EXPERIMENT");
            System.out.println("═".repeat(80));
            System.out.println("Optimizers to Test: " + OPTIMIZER_CONFIGS.length + " configurations");
            System.out.println("  - 4 ADAM configurations");
            System.out.println("  - 4 RMSPROP configurations");
            System.out.println("  - 4 L2/SGD configurations");
            System.out.println("Parameter Matching: All students within ±" + (MAX_PARAM_DEVIATION * 100) + "% of class baseline");
            System.out.println("Replications: " + NUM_REPLICATIONS + " per configuration");
            
            // Calculate total experiments
            int totalExperiments = 0;
            for (int input : inputsize) {
                for (int output : outputsize) {
                    List<int[]> teachers = getMatchingTeachers(input, output);
                    List<int[]> students = getMatchingStudents(input, output);
                    totalExperiments += teachers.size() * students.size() * OPTIMIZER_CONFIGS.length * NUM_REPLICATIONS;
                }
            }
            System.out.println("Total Experiments: " + totalExperiments);
            System.out.println("═".repeat(80) + "\n");

            System.out.println("Starting baseline calculation...");
            
            // PHASE 1: Calculate baseline parameters
            calculateBaselineParameters();
            
            System.out.println("Starting optimizer experiments...\n");
            
            // PHASE 2: Run experiments with each optimizer
            for (OptimizerConfig optimizer : OPTIMIZER_CONFIGS) {
                CURRENT_OPTIMIZER = optimizer;
                
                System.out.println("\n\n" + "═".repeat(80));
                System.out.println("EXPERIMENTAL RUN: " + optimizer.name);
                System.out.println("Optimizer: " + optimizer.type);
                System.out.println("Learning Rate: " + optimizer.learningRate);
                System.out.println("L2 Lambda: " + optimizer.l2Lambda);
                System.out.println("═".repeat(80) + "\n");
                
                runExperiment();
            }

            System.out.println("\n" + "═".repeat(80));
            System.out.println("ALL OPTIMIZER EXPERIMENTS COMPLETED");
            System.out.println("═".repeat(80));
            System.out.println("✓ Tested " + OPTIMIZER_CONFIGS.length + " optimizer configurations");
            System.out.println("✓ All students within ±" + (MAX_PARAM_DEVIATION * 100) + "% of class baseline");
            System.out.println("✓ " + NUM_REPLICATIONS + " replications per configuration");
            System.out.println("═".repeat(80));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void runExperiment() {
        try {
            for (int input : inputsize) {
                for (int output : outputsize) {
                    List<int[]> matchingTeachers = getMatchingTeachers(input, output);

                    if (matchingTeachers.isEmpty()) {
                        System.out.println("No teacher configs for input=" + input + ", output=" + output);
                        continue;
                    }

                    System.out.println("\n" + "=".repeat(60));
                    System.out.println("PROCESSING INPUT=" + input + ", OUTPUT=" + output);
                    System.out.println("Current Optimizer: " + CURRENT_OPTIMIZER.name);
                    System.out.println("Optimizer Type: " + CURRENT_OPTIMIZER.type);
                    System.out.println("=".repeat(60));

                    List<int[]> matchingStudents = getMatchingStudents(input, output);
                    
                    if (matchingStudents.isEmpty()) {
                        System.out.println("No student configs for input=" + input + ", output=" + output);
                        continue;
                    }

                    String classKey = getClassKey(input, output);
                    Integer classBaseline = CLASS_BASELINE_PARAMS.get(classKey);

                    // ITERATE THROUGH EACH TEACHER
                    for (int teacherIdx = 0; teacherIdx < matchingTeachers.size(); teacherIdx++) {
                        int[] T = matchingTeachers.get(teacherIdx);
                        String teacherDepth = getTeacherDepth(T);

                        System.out.println("\n" + "-".repeat(60));
                        System.out.println("TEACHER " + (teacherIdx + 1) + "/" + matchingTeachers.size());
                        System.out.println("Architecture: " + getConfigLabel(T) + " (" + teacherDepth + ")");
                        System.out.println("-".repeat(60));

                        // REPLICATIONS LOOP
                        for (int replication = 0; replication < NUM_REPLICATIONS; replication++) {
                            System.out.println("\n" + "*".repeat(50));
                            System.out.println("REPLICATION " + (replication + 1) + "/" + NUM_REPLICATIONS);
                            System.out.println("*".repeat(50));

                            // Set seed for reproducibility
                            long seed = (teacherIdx * 1000 + replication * 100);
                            
                            // Prepare training and validation datasets
                            fnTrain.Tinputs = new double[numdatapoints][input];
                            fnTrain.Ttargets = new double[numdatapoints][output];
                            fnTrain.TinputsValidate = new double[numValdatapoints][input];
                            fnTrain.TtargetsValidate = new double[numValdatapoints][output];

                            fnTrain.Teacher = new network();
                            fnCreate.createNetworkStd(fnTrain.Teacher, T);

                            // Generate balanced training dataset
                            boolean trainingDataBalanced = false;
                            int trainAttempts = 0;
                            int maxAttempts = 100;

                            while (!trainingDataBalanced && trainAttempts++ < maxAttempts) {
                                trainingDataBalanced = fnTeacher.createDataset(fnTrain.Teacher, fnTrain.Tinputs, fnTrain.Ttargets);
                            }

                            if (!trainingDataBalanced) {
                                System.out.println("WARNING: Failed to generate BALANCED training data after " + maxAttempts + " attempts.");
                                continue;
                            }

                            // Generate balanced validation dataset
                            boolean validationDataBalanced = false;
                            int valAttempts = 0;

                            while (!validationDataBalanced && valAttempts++ < maxAttempts) {
                                validationDataBalanced = fnTeacher.createDataset(fnTrain.Teacher, fnTrain.TinputsValidate, fnTrain.TtargetsValidate);
                            }

                            if (!validationDataBalanced) {
                                System.out.println("WARNING: Failed to generate BALANCED validation data after " + maxAttempts + " attempts.");
                                continue;
                            }

                            int teacherParams = countNetworkParameters(fnTrain.Teacher);

                            System.out.println("\n=== TEACHER NETWORK ===");
                            System.out.println("Architecture: " + getConfigLabel(T));
                            System.out.println("Depth: " + teacherDepth + " (" + (T.length - 2) + " hidden layers)");
                            System.out.println("Total Parameters: " + teacherParams);
                            System.out.println("Seed: " + seed);
                            System.out.println("Optimizer: " + CURRENT_OPTIMIZER.name);
                            System.out.println("========================\n");

                            // Test each student configuration
                            for (int configIndex = 0; configIndex < matchingStudents.size(); configIndex++) {
                                int[] baseConfig = matchingStudents.get(configIndex);

                                System.out.println("\n" + "-".repeat(50));
                                System.out.println("STUDENT " + (configIndex + 1) + "/" + matchingStudents.size() + 
                                                 " | Rep " + (replication + 1) + "/" + NUM_REPLICATIONS);
                                System.out.println("Original: " + getConfigLabel(baseConfig));
                                System.out.println("Optimizer: " + CURRENT_OPTIMIZER.name);
                                System.out.println("-".repeat(50));

                                // Use class baseline for ALL students in this class
                                ArchitectureResult result = findMatchingArchitecture(baseConfig, classBaseline);

                                if (result == null || result.config == null) {
                                    System.out.println(" ERROR: Could not find suitable architecture within ±" + 
                                                     (MAX_PARAM_DEVIATION * 100) + "% of baseline.");
                                    System.out.println("Skipping configuration " + (configIndex + 1) + "\n");
                                    continue;
                                }

                                int[] S = result.config;
                                int finalParams = result.actualParams;

                                String configLabel = getConfigLabel(S);
                                String studentDepth = getStudentDepth(S);
                                
                                // Determine if this is the baseline config for this class
                                boolean isBaseline = (configIndex == 0);

                                System.out.println("\n CONFIGURATION READY");
                                System.out.println("Student Depth: " + studentDepth);
                                System.out.println("Final Architecture: " + configLabel);
                                System.out.println("Final Parameters: " + finalParams + " (Class Baseline: " + classBaseline + ")");
                                System.out.println("Deviation: " + String.format("%.2f%%", Math.abs(finalParams - classBaseline) / (double) classBaseline * 100));
                                System.out.println("Optimizer: " + CURRENT_OPTIMIZER.name);
                                System.out.println("Baseline Student: " + (isBaseline ? "YES ✓" : "NO (matched to baseline)"));
                                System.out.println("-".repeat(50));

                                // Create directory structure with optimizer name
                                String baseDirectoryPath = String.format(
                                    "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\OptimizerExperiments\\%s\\%dinput_%doutput\\Teacher_%s_%s\\Rep%d\\Student_%s_%s\\Config%d\\",
                                    CURRENT_OPTIMIZER.name.replace(" ", "_"),
                                    input, output, 
                                    teacherDepth, getConfigLabel(T).replace(" ", ""),
                                    replication + 1,
                                    studentDepth, configLabel.replace(" ", ""),
                                    configIndex + 1
                                );

                                ensureDirectoryExists(baseDirectoryPath + "dummy.txt");

                                String commonLogFileName = baseDirectoryPath + "common_log.txt";
                                ensureDirectoryExists(commonLogFileName);

                                try (PrintWriter commonWriter = new PrintWriter(new FileWriter(commonLogFileName, true))) {
                                    commonWriter.println("OPTIMIZER EXPERIMENTAL DESIGN");
                                    commonWriter.println("═══════════════════════════════");
                                    commonWriter.println("Optimizer Configuration: " + CURRENT_OPTIMIZER.name);
                                    commonWriter.println("Optimizer Type: " + CURRENT_OPTIMIZER.type);
                                    commonWriter.println("Learning Rate: " + CURRENT_OPTIMIZER.learningRate);
                                    commonWriter.println("L2 Lambda: " + CURRENT_OPTIMIZER.l2Lambda);
                                    commonWriter.println("Beta1: " + CURRENT_OPTIMIZER.beta1);
                                    commonWriter.println("Beta2: " + CURRENT_OPTIMIZER.beta2);
                                    commonWriter.println("Decay Rate: " + CURRENT_OPTIMIZER.decayRate);
                                    commonWriter.println();
                                    
                                    commonWriter.println("Teacher Exp: " + (teacherIdx + 1) + " | Student: " + (configIndex + 1) + " | Rep: " + (replication + 1));
                                    commonWriter.println("Random Seed: " + seed);
                                    commonWriter.println();
                                    
                                    // Parameter matching validation
                                    commonWriter.println("PARAMETER MATCHING VALIDATION");
                                    commonWriter.println("──────────────────────────────");
                                    commonWriter.println("Class Baseline Parameters: " + classBaseline);
                                    commonWriter.println("Current Parameters:        " + finalParams);
                                    commonWriter.println("Difference:                " + (finalParams - classBaseline));
                                    commonWriter.println("Deviation:                 " + 
                                                       String.format("%.2f%%", Math.abs(finalParams - classBaseline) / (double) classBaseline * 100));
                                    commonWriter.println("Is Baseline Student:       " + (isBaseline ? "YES ✓" : "NO"));
                                    
                                    double deviation = Math.abs(finalParams - classBaseline) / (double) classBaseline;
                                    if (deviation <= MAX_PARAM_DEVIATION) {
                                        commonWriter.println("Status:                    ✓ ACCEPTABLE (within " + (MAX_PARAM_DEVIATION * 100) + "%)");
                                    } else {
                                        commonWriter.println("Status:                    ✗ OUT OF RANGE");
                                    }
                                    commonWriter.println();
                                    
                                    commonWriter.println("NETWORK CONFIGURATIONS");
                                    commonWriter.println("──────────────────────");
                                    commonWriter.println("Input Size: " + input);
                                    commonWriter.println("Output Size: " + output);
                                    commonWriter.println("Class Key: " + classKey);
                                    commonWriter.println();
                                    commonWriter.println("Teacher Architecture: " + getConfigLabel(T) + " (" + teacherDepth + ")");
                                    commonWriter.println("Teacher Parameters: " + teacherParams);
                                    commonWriter.println();
                                    commonWriter.println("Student Original: " + getConfigLabel(baseConfig));
                                    commonWriter.println("Student Final: " + configLabel + " (" + studentDepth + ")");
                                    commonWriter.println("Student Parameters: " + finalParams);
                                    commonWriter.println();

                                    fnTrain.Student = new network();
                                    // Use 25% skip connections (you can adjust this)
                                    double skipPercentage = 0.25;
                                    fnCreate.createNetwork(fnTrain.Student, S, skipPercentage, commonWriter);

                                    int studentParams = countNetworkParameters(fnTrain.Student);

                                    commonWriter.println("\n=== PARAMETER VERIFICATION ===");
                                    commonWriter.println("Teacher Parameters: " + teacherParams);
                                    commonWriter.println("Class Baseline:     " + classBaseline);
                                    commonWriter.println("Actual Student:     " + studentParams);
                                    commonWriter.println("Match vs Baseline:  " + 
                                        (Math.abs(studentParams - classBaseline) <= classBaseline * MAX_PARAM_DEVIATION ? "✓ YES" : "✗ NO"));
                                    commonWriter.println();

                                    int teacherLayers = countLayers(fnTrain.Teacher);
                                    int studentLayers = countLayers(fnTrain.Student);

                                    commonWriter.println("Teacher Layers: " + teacherLayers + " | Student Layers: " + studentLayers);
                                    commonWriter.println();

                                    fnCreate.verifyOutputLayerConnections(fnTrain.Student, commonWriter);
                                    fnCreate.verifyOutputLayerConnections(fnTrain.Teacher, commonWriter);
                                } catch (IOException e) {
                                    System.out.println("Error writing to common log file: " + e.getMessage());
                                }

                                // Train with the specified optimizer
                                trainWithOptimizer(baseDirectoryPath, input, output, S, classBaseline, replication + 1);
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Train with the current optimizer configuration
    private static void trainWithOptimizer(String baseDirectoryPath, int input, int output, 
                                         int[] S, int baseline, int replication) {
        String logFileName = baseDirectoryPath + "training_log.txt";
        ensureDirectoryExists(logFileName);

        try (PrintWriter writer = new PrintWriter(new FileWriter(logFileName, true))) {
            writer.println("");
            writer.println("OPTIMIZER TRAINING - " + CURRENT_OPTIMIZER.name);
            writer.println("═══════════════════════════════");
            writer.println("Replication: " + replication);
            writer.println("Date: " + new java.util.Date());
            writer.println("");
            
            writer.println("OPTIMIZER CONFIGURATION");
            writer.println("───────────────────────");
            writer.println("Name: " + CURRENT_OPTIMIZER.name);
            writer.println("Type: " + CURRENT_OPTIMIZER.type);
            writer.println("Learning Rate: " + CURRENT_OPTIMIZER.learningRate);
            writer.println("L2 Lambda: " + CURRENT_OPTIMIZER.l2Lambda);
            writer.println("Beta1: " + CURRENT_OPTIMIZER.beta1);
            writer.println("Beta2: " + CURRENT_OPTIMIZER.beta2);
            writer.println("Decay Rate: " + CURRENT_OPTIMIZER.decayRate);
            writer.println();
            
            writer.println("STUDENT NETWORK");
            writer.println("───────────────");
            writer.println("Input Size: " + input);
            writer.println("Output Size: " + output);
            writer.println("Architecture: " + getConfigLabel(S));
            writer.println("Class Baseline Parameters: " + baseline);
            writer.println();

            // Create student network with 25% skip connections
            double skipPercentage = 0.25;
            fnTrain.Student = new network();
            fnCreate.createNetwork(fnTrain.Student, S, skipPercentage, writer);

            int studentParams = countNetworkParameters(fnTrain.Student);
            double deviation = Math.abs(studentParams - baseline) / (double) baseline * 100;
            
            writer.println("Student Parameters: " + studentParams);
            writer.println("Deviation from Class Baseline: " + String.format("%.2f%%", deviation));
            writer.println();

            // Set optimizer parameters before training
            setOptimizerParameters();
            
            writer.println("TRAINING STARTED");
            writer.println("═════════════════");
            writer.println("Epochs: " + numepochs);
            writer.println("Training Samples: " + numdatapoints);
            writer.println("Validation Samples: " + numValdatapoints);
            writer.println();
            
            long startTime = System.currentTimeMillis();
            
            // Train with the current optimizer
            fnTrain.Train(numepochs, input, output, 0, skipPercentage, writer);
            
            long endTime = System.currentTimeMillis();
            long trainingTime = (endTime - startTime) / 1000;
            
            writer.println();
            writer.println("═══════════════════════════════");
            writer.println("TRAINING COMPLETED");
            writer.println("═══════════════════════════════");
            writer.println("Total Time: " + trainingTime + " seconds");
            writer.println("Time per Epoch: " + String.format("%.2f", trainingTime / (double)numepochs) + " seconds");
            writer.println("Optimizer: " + CURRENT_OPTIMIZER.name);
            writer.println("");
            
            // Log final training metrics
            logTrainingMetrics(writer);
            
        } catch (IOException e) {
            System.out.println("Error writing to training log file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Set optimizer parameters based on current configuration
    private static void setOptimizerParameters() {
        // You need to implement this in your fnTrain class
        // Example implementation:
        // fnTrain.setOptimizer(
        //     CURRENT_OPTIMIZER.type,
        //     CURRENT_OPTIMIZER.learningRate,
        //     CURRENT_OPTIMIZER.l2Lambda,
        //     CURRENT_OPTIMIZER.beta1,
        //     CURRENT_OPTIMIZER.beta2,
        //     CURRENT_OPTIMIZER.decayRate
        // );
        
        System.out.println("Setting optimizer: " + CURRENT_OPTIMIZER.name);
        System.out.println("  Type: " + CURRENT_OPTIMIZER.type);
        System.out.println("  Learning Rate: " + CURRENT_OPTIMIZER.learningRate);
        System.out.println("  L2 Lambda: " + CURRENT_OPTIMIZER.l2Lambda);
    }
    
    // Log final training metrics
    private static void logTrainingMetrics(PrintWriter writer) {
        // You need to implement this based on your training metrics
        // Example:
        writer.println("FINAL TRAINING METRICS");
        writer.println("──────────────────────");
        writer.println("Training Loss: [Implement loss tracking]");
        writer.println("Validation Loss: [Implement validation loss tracking]");
        writer.println("Accuracy: [Implement accuracy tracking]");
        writer.println();
    }

    // Simplified architecture matching
    private static ArchitectureResult findMatchingArchitecture(int[] baseConfig, int baseline) {
        final int MAX_ITERATIONS = 100;
        int minParams = (int) Math.floor(baseline * (1 - MAX_PARAM_DEVIATION));
        int maxParams = (int) Math.ceil(baseline * (1 + MAX_PARAM_DEVIATION));

        System.out.println("\nPARAMETER MATCHING VALIDATION");
        System.out.println("Class Baseline:      " + baseline + " params");
        System.out.println("Acceptable range:    " + minParams + " - " + maxParams + " params");

        // First, test the base configuration as-is (with 25% skip)
        double skipPercentage = 0.25; // Fixed 25% skip for optimizer experiments
        int baseParams = createAndCountParams(baseConfig, skipPercentage);
        if (baseParams >= minParams && baseParams <= maxParams) {
            System.out.println("✓ BASE CONFIGURATION MATCHES!");
            System.out.println("Architecture: " + architectureToString(baseConfig));
            System.out.println("Parameters: " + baseParams);
            return new ArchitectureResult(baseConfig, skipPercentage, baseParams);
        }

        // If base doesn't match, try adjustments
        int[] current = baseConfig.clone();
        
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            int currentParams = createAndCountParams(current, skipPercentage);
            
            if (currentParams >= minParams && currentParams <= maxParams) {
                double deviation = Math.abs(currentParams - baseline) / (double) baseline * 100;
                System.out.println("✓ MATCH FOUND at iteration " + (iteration + 1));
                System.out.println("Architecture: " + architectureToString(current));
                System.out.println("Parameters: " + currentParams);
                System.out.println("Deviation: " + String.format("%.2f%%", deviation));
                return new ArchitectureResult(current, skipPercentage, currentParams);
            }
            
            // Calculate scaling ratio
            double targetRatio = (double) baseline / currentParams;
            
            // Scale ALL hidden layers proportionally
            if (current.length > 2) {
                for (int i = 1; i < current.length - 1; i++) {
                    int newSize = Math.max(2, (int) Math.round(current[i] * targetRatio));
                    current[i] = newSize;
                }
            }
        }
        
        System.out.println("✗ Failed to find matching architecture");
        return null;
    }

    // Helper method to create network and count parameters
    private static int createAndCountParams(int[] config, double skipPercent) {
        try {
            network testNet = new network();
            String tempLog = "C:\\Users\\jimmy\\Documents\\NetBeansProjects\\DNNSG\\temp_param_check.txt";
            ensureDirectoryExists(tempLog);
            
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempLog))) {
                fnCreate.createNetwork(testNet, config, skipPercent, writer);
                return countNetworkParameters(testNet);
            }
        } catch (Exception e) {
            System.out.println("Error creating test network: " + e.getMessage());
            return -1;
        }
    }

    private static List<int[]> getMatchingTeachers(int input, int output) {
        List<int[]> matching = new ArrayList<>();
        for (int[] config : teacherConfigs) {
            if (config[0] == input && config[config.length - 1] == output) {
                matching.add(config);
            }
        }
        return matching;
    }

    private static List<int[]> getMatchingStudents(int input, int output) {
        List<int[]> matching = new ArrayList<>();
        for (int[] config : studentConfigs) {
            if (config[0] == input && config[config.length - 1] == output) {
                matching.add(config);
            }
        }
        return matching;
    }

    private static String getTeacherDepth(int[] config) {
        int hiddenLayers = config.length - 2;
        if (hiddenLayers == 1) return "Shallow";
        if (hiddenLayers == 2) return "Medium";
        if (hiddenLayers == 3) return "Deep";
        return "Custom";
    }

    private static String getStudentDepth(int[] config) {
        int hiddenLayers = config.length - 2;
        if (hiddenLayers <= 3) return "Shallow (≤3L)";
        if (hiddenLayers <= 6) return "Medium (4-6L)";
        return "Deep (7+L)";
    }

    public static int countLayers(network net) {
        int count = 0;
        layernode current = net.inputlayernode;
        while (current != null) {
            count++;
            current = current.next;
        }
        return count;
    }

    public static int countNetworkParameters(network net) {
        int totalParams = 0;
        layernode layer = net.inputlayernode;

        while (layer != null) {
            node n = layer.firstnode;
            while (n != null) {
                if (layer != net.inputlayernode) {
                    totalParams++;
                }
                edge e = n.firstedge;
                while (e != null) {
                    totalParams++;
                    e = e.next;
                }
                n = n.next;
            }
            layer = layer.next;
        }
        return totalParams;
    }

    private static String getConfigLabel(int[] config) {
        StringBuilder label = new StringBuilder();
        for (int i = 0; i < config.length; i++) {
            label.append(config[i]);
            if (i == 0) {
                label.append("n");
            } else if (i == config.length - 1) {
                label.append("o");
            } else {
                label.append("n");
            }
            if (i < config.length - 1) {
                label.append("-");
            }
        }
        return label.toString();
    }

    static class ArchitectureResult {
        int[] config;
        double skipPercentage;
        int actualParams;

        ArchitectureResult(int[] config, double skipPercentage, int actualParams) {
            this.config = config;
            this.skipPercentage = skipPercentage;
            this.actualParams = actualParams;
        }
    }

    public static String architectureToString(int[] config) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < config.length; i++) {
            sb.append(config[i]);
            if (i == 0) sb.append("n");
            else if (i == config.length - 1) sb.append("o");
            else sb.append("n");
            if (i < config.length - 1) sb.append("-");
        }
        return sb.toString();
    }
}