// 爬去哔哩哔哩的弹幕
// r'[\u4e00-\u9fa5]'
const ibili = require('ibili')
const fs = require("fs")
ibili.loadbarrage("https://www.bilibili.com/video/BV1tT4y1G74x?from=search&seid=5539212803983561338").then((data)=>{
    console.log(data.length)
    var barrage = data.map(el=>{
        var text = el.text
        var filter_text = text.replace(/[^\u4e00-\u9fa5]/g, "").replace(/\s/g, "")
        return filter_text
    }).filter(el=>(el !== "" && el.length > 3 && el.length < 20 && !isOneWord(el))).join("\n")
    // console.log(barrage)
    fs.writeFile("test.txt", barrage, ()=>{
        console.log("test.txt is build...")
    })
})

function isOneWord(el){
    var cache = null
    var flag = true
    for(let i=0, len=el.length; i<len; i++){
        if(!cache){
            cache = el[i]
            continue
        }
        if(el[i] !== cache){
            return false
        }
    }
    return flag
}
