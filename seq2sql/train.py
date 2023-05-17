from util.utils import *
from seq2sql.model.seq2sql import Seq2SQL
from util.constants import *
from util.graph_plotter import plot_data


def train_seq2sql():
    # Load training and validation (dev) dataset
    sql_data, table_data, TRAIN_DB = load_dataset("train")
    validation_sql_data, validation_table_data, DEV_DB = load_dataset("test")

    # Load the glove word embeddings
    word_emb = load_word_embeddings('glove/glove.6B.300d.txt')

    # Initialize the target model with the word embeddings
    model = Seq2SQL(word_emb, N_word=300, gpu=GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load the file names for the best models that we find during training
    aggregator_model, selection_model, condition_model = best_model_name()
    
    if __import__('os').path.exists(aggregator_model):
        model.agg_pred.load_state_dict(torch.load(aggregator_model))
        model.sel_pred.load_state_dict(torch.load(selection_model))
        model.cond_pred.load_state_dict(torch.load(condition_model))

    # Initialize the starting values of accuracy by running the model once without training
    init_acc = epoch_acc(model, BATCH_SIZE, validation_sql_data, validation_table_data)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print('Initial dev accuracy: %s\n  breakdown on (agg, sel, where): %s' % init_acc)

    # Save the untrained model as the initial best
    torch.save(model.sel_pred.state_dict(), selection_model)
    torch.save(model.agg_pred.state_dict(), aggregator_model)
    torch.save(model.cond_pred.state_dict(), condition_model)
    
    # Store the losses per epoch for loss curve
    epoch_losses = []

    for i in range(TRAINING_EPOCHS):
        print('Epoch :', i + 1)
        
        # Train the model on training dataset only
        epoch_loss = epoch_train(model, optimizer, BATCH_SIZE, sql_data, table_data)
        epoch_losses.append(epoch_loss)

        print('Loss =', epoch_loss)

        # Check model accuracy on training and validation set
        training_accuracy = epoch_acc(model, BATCH_SIZE, sql_data, table_data)
        print('Train accuracy: %s\n   breakdown result: %s' % training_accuracy)
        
        validation_accuracy = epoch_acc(model, BATCH_SIZE, validation_sql_data, validation_table_data)
        print('Dev accuracy: %s\n   breakdown result: %s' % validation_accuracy)
        
        # If the accuracy is better than the previous best, update the best scores and models
        if validation_accuracy[1][0] > best_agg_acc:
            best_agg_acc = validation_accuracy[1][0]
            best_agg_idx = i + 1
            torch.save(model.agg_pred.state_dict(), aggregator_model)
            torch.save(
                model.agg_pred.state_dict(), 
                aggregator_model.replace(
                    'saved_model', 
                    '/content/drive/MyDrive/Saved_Models'
                )
            )
        if validation_accuracy[1][1] > best_sel_acc:
            best_sel_acc = validation_accuracy[1][1]
            best_sel_idx = i + 1
            torch.save(model.sel_pred.state_dict(), selection_model)
            torch.save(
                model.sel_pred.state_dict(), 
                selection_model.replace(
                    'saved_model', 
                    '/content/drive/MyDrive/Saved_Models'
                )
            )
        if validation_accuracy[1][2] > best_cond_acc:
            best_cond_acc = validation_accuracy[1][2]
            best_cond_idx = i + 1
            torch.save(model.cond_pred.state_dict(), condition_model)
            torch.save(
                model.cond_pred.state_dict(), 
                    condition_model.replace(
                        'saved_model', 
                        '/content/drive/MyDrive/Saved_Models'
                    )
            )

        print('Best val accuracy = %s, on epoch %s individually' % (
            (best_agg_acc, best_sel_acc, best_cond_acc),
            (best_agg_idx, best_sel_idx, best_cond_idx)))

    # save epoch vs loss graph
    plot_data(x = range(TRAINING_EPOCHS), y = epoch_losses, xlabel = "Epochs", ylabel = "Loss", label = "Loss Graph for target seq2sql model")


if __name__ == '__main__':
    train_seq2sql()
