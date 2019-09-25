from utils.calibration import *
from utils.loss import *
from utils.help import *

def main():

    #Input:
    nb_experiments = 2
    dataset_name, model_name = "cifar100", "densenet40"
    metrics = ["acc", "nll", "ece", "brier"]
    methods = ["Uncalibrated", "Temperature-Scaling", "UTS"]

    #Criterions:
    acc_criterion = ACCLoss()
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss()
    brier_criterion = BrierLoss()

    print("\n========================================")
    for metric in metrics:
        for method in methods:

            final_result = MeanStdInfo()
            for i in range(nb_experiments):
            
                logits, labels = load_logits_labels(dataset_name, model_name)
                v_logits, v_labels, t_logits, t_labels = data_split(logits, labels, test_size=0.8)
        
                #Calculate temperature
                if method == "Temperature-Scaling":
                    model_TS = TS()
                    t_TS = model_TS.find_best_T(v_logits, v_labels)
                    t_logits /= t_TS
                elif method == "UTS":           
                    model_UTS = UTS()
                    t_UTS = model_UTS.find_best_T(v_logits, v_labels)
                    t_logits /= t_UTS

                #Generate results:
                if metric == "acc":
                    result = acc_criterion(t_logits, t_labels).item()
                elif metric == "nll":
                    result = nll_criterion(t_logits, t_labels).item()
                elif metric == "ece":
                    result = ece_criterion(t_logits, t_labels).item()
                elif metric == "brier":
                    result = brier_criterion(t_logits, t_labels).item()
                final_result.update(result)
            
            #Output results:
            print(method, metric + ": ", str(final_result))
        print("========================================\n")
        
if __name__== "__main__":
    main()
