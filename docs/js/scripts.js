function generate_row_header(array){
    var row = document.createElement("tr");
    for(var i=0; i<4; i++){
        var cell = document.createElement("th");
        cell.setAttribute("scope", "col");
        var cell_text = document.createTextNode(array[i]);
        cell.appendChild(cell_text);
        row.append(cell);
    }
    return row;
}
function generate_row(array){
    var row = document.createElement("tr");
    for(var i=0; i<4; i++){
        var cell;
        if(i==0){
            cell = document.createElement("th");
            cell.setAttribute("scope", "row");
        }else{
            cell = document.createElement("td");
        }
        
        var cell_text = document.createTextNode(array[i]);
        cell.appendChild(cell_text);
        row.append(cell);
    }
    return row;
}

function generate_table(container,
                        inputs,
                        outputs,
                        parameters = []
                        ){
    
    var container = document.getElementById(container);
    
    var tbl = document.createElement("table");
    tbl.className = "table";

    var tblThead = document.createElement("thead");
    tblThead.className = "thead-light";
    var header_row = generate_row_header(["Name", "Device", "Type", "Size"]);
    tblThead.appendChild(header_row);
    tbl.appendChild(tblThead);

    var tblHInputs = document.createElement("thead");
    tblHInputs.className = "thead-dark";
    var header_inputs = generate_row_header(["Inputs", "", "", ""]);
    tblHInputs.appendChild(header_inputs);
    tbl.appendChild(tblHInputs);
    
    var tblInBody = document.createElement("tbody");
    for(var i=0; i<inputs.length; i++){
        var row = generate_row([inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]]);
        tblInBody.appendChild(row);
    }
    tbl.appendChild(tblInBody);
    
    var tblHOutputs = document.createElement("thead");
    tblHOutputs.className = "thead-dark";
    var header_outputs = generate_row_header(["Outputs", "", "", ""]);
    tblHOutputs.appendChild(header_outputs);
    tbl.appendChild(tblHOutputs);
    
    var tblOutBody = document.createElement("tbody");
    for(var i=0; i<outputs.length; i++){
        var row = generate_row([outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]]);
        tblOutBody.appendChild(row);
    }
    tbl.appendChild(tblOutBody);

    if(parameters.length>0){
        var tblHParams = document.createElement("thead");
        tblHParams.className = "thead-dark";
        var header_outputs = generate_row_header(["Parameters", "", "", ""]);
        tblHParams.appendChild(header_outputs);
        tbl.appendChild(tblHParams);
        
        var tblParBody = document.createElement("tbody");
        for(var i=0; i<parameters.length; i++){
            var row = generate_row([parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3]]);
            tblParBody.appendChild(row);
        }
        tbl.appendChild(tblParBody);
    }

    container.appendChild(tbl);
}

function copyFileContents(src, dst){
    var srcObj = document.getElementById(src);
    var dstObj = document.getElementById(dst);
    var doc = srcObj.textContent;
    console.log("Loaded");
    console.log(srcObj);
    
    console.log(doc);
    var code_obj = document.createElement("code");
    code_obj.className = "python";
    var code_text = document.createTextNode("from TorchProteinLibrary import FullAtomModel");
    code_obj.appendChild(code_text);
    dstObj.appendChild(code_obj);
    // console.log(src);
    // console.log(src.dataset);

}

