import mpld3
class SliderView(mpld3.plugins.PluginBase):
    """ Add slider and JavaScript / Python interaction. """

    JAVASCRIPT = """
    mpld3.register_plugin("sliderview", SliderViewPlugin);
    SliderViewPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    SliderViewPlugin.prototype.constructor = SliderViewPlugin;
    SliderViewPlugin.prototype.requiredProps = ["idline", "callback_func"];
    SliderViewPlugin.prototype.defaultProps = {}

    function SliderViewPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    SliderViewPlugin.prototype.draw = function(){
      var line = mpld3.get_element(this.props.idline);
      var callback_func = this.props.callback_func;

      var div = d3.select("#" + this.fig.figid);

      // Create slider
      div.append("input").attr("type", "range").attr("min", 0).attr("max", 10).attr("step", 0.1).attr("value", 1)
          .on("change", function() {
              var command = callback_func + "(" + this.value + ")";
              console.log("running "+command);
              var callbacks = { 'iopub' : {'output' : handle_output}};
              var kernel = IPython.notebook.kernel;
              kernel.execute(command, callbacks, {silent:false});
          });

      function handle_output(out){
        //console.log(out);
        var res = null;
        // if output is a print statement
        if (out.msg_type == "stream"){
          res = out.content.data;
        }
        // if output is a python object
        else if(out.msg_type === "pyout"){
          res = out.content.data["text/plain"];
        }
        // if output is a python error
        else if(out.msg_type == "pyerr"){
          res = out.content.ename + ": " + out.content.evalue;
          alert(res);
        }
        // if output is something we haven't thought of
        else{
          res = "[out type not implemented]";  
        }

        // Update line data
        line.data = JSON.parse(res);
        line.elements()
          .attr("d", line.datafunc(line.data))
          .style("stroke", "black");

       }

    };
    """

    def __init__(self, line, callback_func):
        self.dict_ = {"type": "sliderview",
                      "idline": mpld3.utils.get_id(line),
                      "callback_func": callback_func}