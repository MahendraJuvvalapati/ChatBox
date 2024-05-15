package chatBox;

import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.BertQaInput;
import ai.djl.modality.nlp.qa.BertQaOutput;
import ai.djl.translate.TranslateException;

public class ChatBox extends HttpServlet {

    private static final String MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad";

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String query = request.getParameter("query");
        // Process the user's query and generate response
        String responseText = generateResponse(query);
        
        // Set response content type
        response.setContentType("text/plain");
        // Get the response writer
        PrintWriter out = response.getWriter();
        // Write response
        out.println(responseText);
    }

    private String generateResponse(String query) {
        try (Model model = Model.newInstance(MODEL_NAME)) {
            Criteria<BertQaInput, BertQaOutput> criteria = Criteria.builder()
                    .setTypes(BertQaInput.class, BertQaOutput.class)
                    .optModel(model)
                    .optTranslator(new BertQaTranslator())
                    .build();
            try (ZooModel<BertQaInput, BertQaOutput> zooModel = ModelZoo.loadModel(criteria);
                 Predictor<BertQaInput, BertQaOutput> predictor = model.newPredictor(zooModel)) {
                BertQaInput input = new BertQaInput(query, "What is the answer?");
                BertQaOutput output = predictor.predict(input);
                return output.getAnswer();
            } catch (TranslateException e) {
                e.printStackTrace();
                return "I'm sorry, I couldn't understand that.";
            }
        } catch (ModelException | IOException e) {
            e.printStackTrace();
            return "I'm sorry, I couldn't understand that.";
        }
    }
}
