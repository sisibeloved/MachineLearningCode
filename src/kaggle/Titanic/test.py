# Import libraries necessary for this project
import pandas as pd

# Load the dataset
in_file = 'test.csv'
full_data = pd.read_csv(in_file)
# data = full_data.drop('Survived', axis = 1)


def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """

    predictions = []
    for _, passenger in data.iterrows():

        # Remove the 'pass' statement below
        # and write your prediction conditions here
        if passenger['Sex'] == 'female':
            if passenger['Pclass'] == 3:
                if passenger['SibSp'] > 0 and passenger['Parch'] > 0:
                    predictions.append(0)
                else:
                    predictions.append(1)
            else:
                predictions.append(1)
        elif passenger['Age'] < 10:
            predictions.append(1)
        else:
            predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)


# Make the predictions
predictions = predictions_3(full_data)

result = pd.DataFrame({'PassengerId': full_data['PassengerId'], 'Survived': predictions})
result.to_csv('result.csv', index=False)
