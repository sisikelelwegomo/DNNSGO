package dnnsgo;

public class TrainingContext {
    public double[][] inputs;
    public double[][] targets;
    public double[][] inputsValidate;
    public double[][] targetsValidate;
    public network teacher;
    public network student;

    public TrainingContext(double[][] inputs, double[][] targets,
                           double[][] inputsValidate, double[][] targetsValidate,
                           network teacher, network student) {
        this.inputs = inputs;
        this.targets = targets;
        this.inputsValidate = inputsValidate;
        this.targetsValidate = targetsValidate;
        this.teacher = teacher;
        this.student = student;
    }
}
