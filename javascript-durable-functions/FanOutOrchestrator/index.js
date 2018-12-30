const df = require("durable-functions");

module.exports = df.orchestrator(function*(context){
    context.log("Starting fan out sample");
    const tasks = [];
    tasks.push(context.df.callActivity("HelloActivity", "Tokyo"));
    tasks.push(context.df.callActivity("HelloActivity", "Seattle"));
    tasks.push(context.df.callActivity("HelloActivity", "London"));

    const results = yield context.df.Task.all(tasks);

    return results.join(",");
});