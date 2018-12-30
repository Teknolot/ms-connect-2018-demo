const df = require("durable-functions");

module.exports = df.orchestrator(function*(context){
    context.log("Starting chain function sample");
    const output = [];
    output.push(yield context.df.callActivity("HelloActivity", "Tokyo"));
    output.push(yield context.df.callActivity("HelloActivity", "Seattle"));
    output.push(yield context.df.callActivity("HelloActivity", "London"));

    return output;
});