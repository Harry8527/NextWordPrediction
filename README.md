# Overview:
It is a built a lightweight next word prediction NLP model. This model will take an incomplete sentence as an input, and predict the next word after the last word using GPT-2

------------------------------------------------------ Code Snippet Explanation ---------------------------------------------------------------------------------------------

# Import libraries
    # from transformers import GPT2LMHeadModel, GPT2Tokenizer

        We are importing GPT2LMHeadModel, and GPT2Tokenizer class from the hugging face transformers library.

        GPT2LMHeadModel 
            This is a pre-trained language model based on GPT-2, specifically fined-tuned for language modelling tasks like word-prediction.

        GPT2Tokenizer
            It is a tokenizer for GPT2 model, which converts raw-text(input sentence in english) to input IDS(numbers) that the model can understand, and
            vice-versa. Each model on hugging face comes with its own tokenizer for consistent tokenization.

            # Tokenization
                It is a process of convert raw text(written in english provided by the user, or in any other language) into tokens(like subwords or words), and
                then mapping those tokens to numerical IDs(integers that the model can understand). 
                i.e. Convert text -> tokens
                    tokens -> numerical IDs.
                For example: Let's say our input is: "I love AI". So, when we perform tokenization then:
                    "I love AI" -> ["I", "love", "AI"]   :: raw text -> tokens
                    ["I", "love", "AI"] -> [1234, 4554, 1232]  :: tokens -> numerical IDs.
        

    # import torch
        We are importing pytorch which is a deep learning framework. It is used in this script to convert the token IDs received as an output from tokenizer to 
        pytorch tensors. It is also used to perform all the mathematical calculations needed to run the model. Even though we are using Hugging Face's high-level 
        interface, all the actual processing - like turning token IDs to tensors, and performing predictions is handled by pytorch internally.


# model_name = "gpt2"
    We declared a variable named "model_name", and assign the string "gpt2" to it. This variable will later be  passed as an argument to from_pretrained() 
    of GPT2Tokenizer and GPT2LMHeadModel class. This way we will tell:
     1. The tokenizer to load the the vocabulary, and tokenization rules of GPT-2 model.
     2. The Head model to load the pre-trained model, which is fine-tuned for next word-prediction. 


# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    This is where we are actually loading the gpt2 model from the GPT2Tokenizer class using from_pretrained() by passing the name of the model as an argument
    to it. Now, we will have all the tokenization rules of the gpt2 model.Once loaded, the tokenizer can be used to:
        1. Convert the raw text(user input in english) to Input Ids(numerical representation of tokens) which the model understands.
        2. Later convert the predicted output of the model(token IDs), back to raw text. (which the user can understand)
    This is a crucial step because models like GPT-2 don't work directly with plain text - they only understand numbers(token IDs)

# model = GPT2LMHeadModel.from_pretrained(model_name)
    This line loads the pre-trained gpt2 language model, and store its reference in the model variable. 
    1. The model is already trained on the large amount of text data,so it has learned the partterns of natural language.
    2. We don't need to train or fine-tune this model again before making next word output predictions. 
        In this script, we will directly use this model to make the predictions of most likely next word(s) based on the user input.

        # Brief about what the reference is actually pointing to:
            We are storing a fully-initialized instance of GPT-2 model in the model variable. What that means is that, we are storing the entire model
            architecture(i.e. its configuration, all the layers, and weights learned during training)

# model.eval()
    We are setting our model in evaluation mode. This is the mode in which based on user input our model predicts output.This is important because certain neural network
    models behaves differently in training and evaluation mode.

    There are commonly 2 modes we set for a model:
    1. train():
        a. This is set when we are training the model.
        b. It enables training-specific behavior of the model such as adjusting weights, and applying dropouts.
        c. To set a model in train mode, we write as: model.train()
    2. eval():
        a. This mode is set, when we are using the model for predicting the output based on input.
        b. Disable training behavior like weight adjustments and dropouts.
        c. Ensure consistent results during inference.(inferencing means that model is being used to generate predictions, and not for training.)
        d. To set the model in evaluating mode, we write as: model.eval()

        # Dropout
            It is a regularization technique used during training to prevent overfitting by randomly setting a fraction of the neurons' outputs to
            zero at each training step.
            Example
            Suppose our input data is getting forward passed through 6 neurons everytime, and then the output is generated.
            So when dropout is enabled and set to 0.5 (as nn.Dropout(p-0.5)) then in every training step half of the neurons are turned OFF(i.e., set
            to 0 or not considered or skipped) for generating the final prediction. This way the model can't memorize the training data.
            (which in technical terms is also known as overfitting). So, we can also say that using Dropout technique
            we prevent overfitting of the model.

# tokenizer.pad_token = tokenizer.eos_token
    We are setting the "pad_token" variable of our GPT2 tokenizer to be the same value as its "eos_token".
    Since GPT-2 was not trained with a padding token, we manually assign the end-of-sequence token('eos_token') to be used wherever padding is needed.
    This is majorly useful in batch processing where all the input sequences must be of the same length.

    "eos_token" has a special meaning and value in the GPT2Tokenizer class. As the word suggests, it indicates the end of sentence to the tokenizer.
    The default value of the eos_token  is ''(an empty string), and its token-id is 50256 in GPT-2. For shorter sentences in a batch our tokenizer
    will use eos_token for padding.
    
    # Usecase 
        This is especially useful when we are passing a batch of sentences to a model for next word prediction. Our model expects all the inputs sequences 
        to be of same length, but this is not usually the case. 
        For example: 
            sentences = [
                           "I love artificial intelligence",
                            "Can you help me with",
                            "The weather today is"
                        ]
            For the above case:
            1. we have 3 input sentences(or our batch size is 3). 
            2. The max_size of the sequence is: 5("Can you help me with"). 
            3. When this batch is passed to the model for tokenization as: inputs = tokenizer(sentences, return_tensors="pt", padding=True), then our tokenizer will
            pad 2 shorter sentences with eos_token to make sure all the sentences in a batch are of the same length.

# model.config.pad_token_id = model.config.eos_token_id
    We are setting pad_token_id of the model to the same value as its eos_token_id.
    Since GPT-2 doesnot have a dedicated padding token by default, we assign the eos_token_id(50286) to serve as the padding token id.
    This is useful during the token generation or inference, as this will make sure that the model can correctly handle sequences that have been paded.

    We already configured the tokenizer to use eos_token for padding(tokenizer.pad_token = tokenizer.eos_token). Now, we are also telling the model itself that the
    padding token has ID 50256, by setting - model.config.pad_token_id = model.config.eos_token_id
    Without this statement, there is a possibility that the model may behave unpredictably during output generation(eg: it might continue predicting even after
    the padded parts of the sentence)

    For example:
        sentences = [
                           "I love artificial intelligence",
                            "Can you help me with",
                            "The weather today is"
                    ]
        For the above case our model will pad the 2 shorter sentences ("I love artificial intelligence", "The weather today is") with pad_token i.e. eos_token.
        
        This will make sure that all the sentences in the input batch are of the same length. And when the tokenization happens as:
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        The shorter sentences will be padded with token ID 50256, so that all the sequences match the length of the longest one.
        
        Now, at the time of generating the prediction  (output_ids = model.generate(....) ) when our model encounters eos_token_id value(or 50256), it will know that, 
        its the end of the sentence, and it will ignore this value of the sequence for consideration to tokenize.

# def predict_next_word(prompt_text, max_new_tokens=5):
    We have defined a function named as predict_next_word, which takes the following input parameters:
    1. prompt_text: This is a string input received from the user. Its a partial sentence for which we want the model to predict the next word.
    2. max_new_tokens: This control how many new tokens the model is allowed to generate. By default, the model is allowed to generate a maximum of 5 new tokens
    beyond the last word of the sentence received as user input in prompt_text.
    Sometimes the model generate a phrase instead of a next word, we dont want to truncate the meaningful phrase, and that is the reason we assign max_new_tokens a value
    greater than 1. Even though the model generated a phrase instead of a word, we return the first word only(i.e. the next word or the first word of the phrase) from the
    generated output in this function. 

# inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    Through this statement we are tokenizing the input string prompt_text using the GPT2 tokenizer. Following steps are performed here:
        1. Tokenization : The tokenizer splits the input string into smaller units called tokens based on GPT-2 vocabulary.
        2. Conversion to token IDs: These tokens are then converted to token IDs (which are their corresponding integer value based on GPT2) that the model understands. 
        3. Return as Pytorch tensors: The return_tensors="pt" argument tells the tokenizer to return the token IDs(and other required inputs like attention mask) as Pytorch tensors,
           which can we passed directly to the model for prediction.
        4. padding=True(when needed): This is useful when the user passes multiple sentences as input. Then, padding=True ensures that all shorter sentences are padded to make their
           length equal to the longest sentence in the batch. Since GPT-2 doesnot have a default pad_token, we explicitly set it earlier using:
           tokenizer.pad_token = tokenizer.eos_token
           Here, the padding is done with eos_token_id(which is 50256 in GPT-2) 

           attention_mask: 
                They are useful to inform the model about which token_ids to pay attention to and which to ignore for output prediction. 
                It is mainly useful in case of batch inputs where we have multiple sequences which include padding. To make the length of all sentences(sequences) equal to the longest
                sentence in the batch we perform padding. We also dont want our model to consider padded tokens(eos_token_id or 50256) as meaningful inputs(tokens) while generating the
                output prediction. So in those cases, we use attention mask to indicate which positions are padded tokens(attention_mask value = 0), and which are actual tokens
                (attention_mask value = 1).

                For example: 
                sentences = [
                        "Hello there",
                        "How are you"                            
                ]
                Assume the tokenizer returns:
                {
                    'input_ids': tensor([[1234, 2345, 50256], [4343, 3423, 2342]]),
                    'attention_mask': tensor([[1,1,0], [1,1,1]])
                }
                Notice that for sequence1(Hello there) the 3rd item in the input_ids list is eos_token_id(50256), and the corresponding attention_mask value for that item is 0, which tells
                the model to not consider this value for output prediction, as its a pad_token.

                Note: The length of the attention mask is always same as the length of input sequence. Basically, for every word in the sequence(or for every token Id) we have a corresponding
                attention mask value. It contains only 1s and 0s as:
                - 1 -> its a real token (consider it for output prediction)
                - 0 -> its a pad token (donot consider it for output prediction)

# with torch.no_grad():
    We are instructing the Pytorch autograd engine(which tracks the operations on tensors) to apply the following rules within the with block.
        1. Donot compute the gradients,which are required only during training the model for updating the weight parameters.
        2. Donot store the intermediary tensor values, as we are not going to perform backpropagation, and will only do forward pass(inference).
        3. This way we save computational memory(as we are not storing additional tensors values), and also improve the model performance(as we donot have to perform additional operations for 
        tracking intermediary tensor values)

#  output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50,top_p=0.95, temperature=0.8, pad_token_id=model.config.pad_token_id)
    This is where the actual prediction of the next word is happening. Below is the brief about each of the argument along with its use:
    1. model.generate(): Calling the generate() of the model to make predictions of next word.
    2. **inputs: We are passing the tokenized input(as Pytorch tensors) in the form of a batch which includes input_ids and attention_mask.
    3. max_new_tokens=max_new_tokens: We are limiting the number of new tokens that can be generated in output prediction. This is set to the value of max_new_tokens argument whose value is
    received from the user when predict_next_word() is called. We set this to a value > 1 (default value is 5). So the model can generate meaningful phrases.However from predict_next_word(),
    we only return the first word.
    4. do_sample = True: Enables sampling instead of greedy decoding. In other words, we are telling the model  to do output prediction from a sample of values,  instead of always picking 
    the most obvious next word. This allows the model to perform more diverse and creative output prediction. This also make the predicted output less repetitive for the same set of input.
    5. top_k = 50: Asking the model to make output prediction from the top  50 words of the sample. This way we are limiting the range of predicted output under 50 most
    obvious next word. This helps reduce randomness, while also keeping some creativity.
    6. top_p = 0.95(nucleus sampling): Asking the model to select a subset of words from the sample of 50 most obvious next words, whose combined probability is 95% or above. 
    7. temperature = 0.8: This argument decides the randomness or the creativity level of output prediction. Lower value indicates less creativity. 
    For example, our input string is: "There is a tree in the backyard of my"
        Output:
            If temperature is 0.1: house
            If temperature is 0.8: city
            If temperature is 1.5: universe
        Its value can range between 0.0 to 2.0.
    8. pad_token_id: Ensures that, if the input includes padding(incase of multiple sequences in a batch), which token ID(i.e. 50256) to treat as pad_token at the time of output prediction.
    9. The resultant value of prediction by model.generate(), which could be a phrase or just a word, is stored in output_ids variable. The output_ids is a tensor of token IDs.

# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    We are decoding our tokenized output to human-readable text. As told above, output_ids is a tensor of token IDs, so to convert these tokens into words we will use the tokenizer object 
    of GPT2 class, as this tokenizer will only know the correct mapping of each integer value(or token ID) to word.  The generated output is in the form of batch, within which we can have 
    single  or multiple sequences depending on the number of input sequences/sentence passed at the time of prediction.  As we are passing only 1 sentence as input, that's why at the time 
    of decoding we wrote as output_ids[0]. 
    output_ids[0]: Grabs the first sequence from the batch.
    Each element within the sequence will include the token IDs of input  as well as output. For example, assume:
    Our input prompt is: Hello, 
    output_ids will be: tensor([[15496,    11,   257,   983,   326,   314,   711]]), where the first two token ID elements 15946 and 11 represent input prompt. 
    which maps to: Hello, a game that I play
    where, Hello, is the part of input string.
    Note: The decode result is a continuation of your input.

    The argument skip_special_tokens=True will skip decoding of any special token added by the model. This also removes special tokens like <|endoftext|> that might appear in the output, 
    making the final output text clean. Another example is "`<",  which indicates the end of text in GPT-2

#   new_text = generated_text[len(prompt_text):].strip()
    We are slicing the generated_text string value in such a manner that it starts after the user's input prompt_text and goes till the end of the generated_text string. So, that  we only get
    the newly generated part. We are also removing any additional whitespace from start and end of the sliced string using strip().
    Finally, we are storing that sliced string after removing whitespaces from the start the end into new_text variable.

#   next_word = new_text.split()[0] if new_text else ""
    Following activities are happening here:
    1. new_text.split() : This will split the "new_text" string into a list of words based on whitespace.
    2. new_text.split()[0] : Now we are picking the first word from the list.
    3. if new_text: This checks the value of new_text variable, if its non-empty then the if condition PASS, and the first word of the list(new_text[0]) will get stored in next_word variable.
    4. If new_text is empty, the else block will be executed, and an empty string - "" will be stored in next_word variable.
    NOTE: This line of code ensures that, we return only the first word of prediction from predict_next_word(), even though our model predicts a phrase.

#   return next_word
    Returning the first word predicted by the model from predict_next_word() using the return keyword.

#   while True:
    This will run the prediction_model continuously, until the user type quit or exit as input string.  More details about how the models knows to stop when "quit" or "exit" is typed as
    input string is shared ahead.

#   prompt = input("\nEnter a sentence: ")
    When we run this python file, "Enter a sentence" message will be printed at the console, and the input () will store whatever string user types into prompt variable.

#   if prompt.strip().lower() in ["exit", "quit"]:
        break
    Following activities are happening here:
    1. prompt.strip().lower() : Removes any whitespace at the start and end from the user's input prompt string, and converts it to lowercase letters.
    2. if prompt.strip().lower() : The cleaned string is then compared against "exit" and "quit" word. 
    3. If it matches, the 'break' statement is executed, which stops the while loop and ends the program.

#   suggestion = predict_next_word(prompt)
    When the user input is anything other than "exit" or "quit", that string value will be passed to predict_next_word() as a parameter, and that function will predict and return the
    next_word after it. The returned value will be stored in suggestion variable.

#   print("Next word suggestion:", suggestion)
    This statement will prints the next word(stored in suggestion variable) after user-input predicted by the model to the console, as output for the user.