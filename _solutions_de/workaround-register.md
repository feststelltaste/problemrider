---
title: Workaround-Register
description: Erfassung jedes Workarounds in dem Moment, in dem er
  eingeführt wird — was er verbirgt, was er kostet und was ihn
  überflüssig machen würde — sodass temporäre Fixes nicht unbemerkt
  dauerhaft werden.
category:
- Code
- Process
- Operations
problems:
- accumulation-of-workarounds
- workaround-culture
- increased-technical-shortcuts
- invisible-nature-of-technical-debt
- partial-bug-fixes
- increased-manual-work
- quality-compromises
- hidden-dependencies
- constant-firefighting
- delayed-bug-fixes
- high-technical-debt
- operational-overhead
layout: solution
lang: de
en_slug: workaround-registry
related_solutions:
- slug: blameless-postmortems
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: technical-debt-backlog
  similarity: 0.7
- slug: knowledge-sharing-practices
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
- slug: knowledge-base
  similarity: 0.65
---

## Description

Ein Workaround-Register ist eine einzige, gepflegte Liste der kompensierenden Maßnahmen, von denen ein System abhängt: der manuelle Schritt, den jemand jeden Monat durchführt, der Retry, der eine unzuverlässige Schnittstelle verbirgt, der Sonderfall für einen Kunden, der geplante Job, der Daten repariert, die ein anderer Job beschädigt. Jeder Eintrag protokolliert, welches Problem er kompensiert, was seine Pflege kostet, was benötigt würde, um ihn zu entfernen, und wer ihn wann eingeführt hat. Workarounds sind einzeln rational — unter Zeitdruck ist Kompensieren üblicherweise korrekt — und kollektiv zersetzend, weil jeder von außen unsichtbar ist und jeder einschränkt, was später geändert werden kann. Das Register entmutigt Workarounds nicht. Es macht sie zählbar, sodass die Organisation das angesammelte Gewicht von Entscheidungen sehen kann, die jede isoliert vernünftig erschien.

## How to Apply ◆

> Die teuersten Workarounds in einem Legacy-System sind üblicherweise die außerhalb des Codes: der manuelle Abgleich, die Tabellenkalkulation, die tägliche Prüfung, die jemand so lange durchgeführt hat, dass sich niemand mehr erinnert, dass es ein Workaround ist.

- **Protokollieren Sie den Workaround in dem Moment, in dem er erstellt wird**, als Teil der Änderung, die ihn einführt. Rückwirkende Archäologie findet nur einen Bruchteil davon, und die Person, die ihn eingeführt hat, ist die Einzige, die weiß, was er kompensiert.
- Erfassen Sie fünf Felder und nicht mehr: **was er tut, welches Problem er verbirgt, was seine Pflege kostet, was ihn entfernen würde, und wann er eingeführt wurde.** Mehr Felder bedeuten weniger Einträge, und Abdeckung zählt mehr als Detail.
- Beziehen Sie **betriebliche und organisatorische Workarounds** ein, nicht nur Code. Der manuelle Schritt, die Tabellenkalkulation, die wiederkehrende Kalendererinnerung und der Runbook-Eintrag, der sagt "wenn das passiert, tue das" gehören alle dazu. Diese sind häufig die größten Kosten und die am wenigsten sichtbaren.
- **Markieren Sie den Workaround im Code selbst** mit einem konsistenten, grep-baren Marker, der zum Register-Eintrag verlinkt. Ein Entwickler, der auf den Code stößt, muss herausfinden können, warum er da ist, und ein Kommentar, der "temporär" sagt, ohne Datum oder Referenz, ist schlimmer als nichts.
- **Überprüfen Sie das Register in fester Taktung** — vierteljährlich ist üblicherweise richtig. Das Review stellt zwei Fragen pro Eintrag: Ist das kompensierte Problem noch real, und hat sich der Aufwand geändert. Workarounds überleben regelmäßig ihre Ursache vollständig, und diese zu finden ist die günstigste mögliche Bereinigung.
- **Speisen Sie Entfernungskandidaten in das Verbesserungsbudget ein.** Das Register lohnt sich nur zu pflegen, wenn Einträge es gelegentlich verlassen, und diejenigen zu priorisieren sind die mit hohen Wartungskosten und niedrigen Entfernungskosten.
- **Berichten Sie die Anzahl und den Trend** neben anderen Gesundheitsmaßen. Ein stetig wachsendes Register ist ein System, das Einschränkungen ansammelt, und der Trend ist eine überzeugendere Zahl für das Management als jeder einzelne Eintrag.
- **Nutzen Sie es nicht zur Schuldzuweisung.** Ein Register, das genutzt wird, um zu identifizieren, wer Abkürzungen genommen hat, wird innerhalb eines Monats aufhören, ausgefüllt zu werden, und die Workarounds werden fortbestehen, während die Aufzeichnung von ihnen verschwindet.
- **Setzen Sie Ablaufdaten**, wo der Workaround genuin als temporär gemeint ist, und lassen Sie den Ablauf ein Review auslösen statt eine automatische Entfernung. Ein angegebenes Datum, das verstreicht, ist zumindest eine sichtbare Entscheidung, ihn zu behalten.

## Tradeoffs ⇄

> Das Register verwandelt unsichtbare angesammelte Einschränkung in eine zählbare Liste, auf Kosten von Disziplin bei der Pflege und dem Unbehagen einer expliziten Aufzeichnung von Kompromiss.

**Vorteile:**

- Workarounds hören auf, unsichtbar zu sein, was die Eigenschaft ist, die es ihnen erlaubt, sich unbegrenzt anzusammeln, ohne dass je eine Entscheidung über sie getroffen wird.
- Entwickler, die auf unerklärten Code stoßen, können herausfinden, warum er existiert, was den häufigen Fehler verhindert, einen Workaround zu entfernen und das Problem, das er verbarg, wieder einzuführen.
- Veraltete Workarounds werden gefunden. Ein erheblicher Anteil kompensiert Probleme, die vor Jahren behoben wurden, oder Systeme, die ausgemustert wurden.
- Der Trend gibt ein Frühsignal von Degradation, oft früher als Defektraten oder Zykluszeiten.
- Die echten Kosten aufgeschobener Fixes werden sichtbar, was eines der wenigen wirksamen Argumente ist, ein zugrunde liegendes Problem zu adressieren, statt es erneut zu kompensieren.

**Kosten und Risiken:**

- Register verfallen. Eines, das nicht überprüft wird, wird zu einer veralteten Liste, der niemand vertraut, was schlimmer ist als keine, weil es falsches Vertrauen schafft, dass Workarounds verfolgt werden.
- Das Protokollieren ist genau dann leicht zu überspringen, wenn es am meisten zählt — unter dem Zeitdruck, der den Workaround überhaupt erst produzierte.
- Eine explizite Liste von Kompromissen kann von einem unsympathischen Leser gegen das Team genutzt werden, sodass sie als Ingenieursinstrument statt als Geständnis gerahmt werden muss.
- Die Identifikation von Nicht-Code-Workarounds erfordert Kooperation von Fachabteilungen, die ihren manuellen Schritt möglicherweise überhaupt nicht als Workaround wahrnehmen.
- Workarounds sichtbar zu machen, ohne Kapazität zur Entfernung bereitzustellen, produziert Frustration und eine wachsende Liste, die nur demonstriert, dass nichts getan wird.

## How It Could Be

Ein Team, das ein Krankenhausabrechnungssystem pflegte, begann, Workarounds zu protokollieren, nachdem ein Vorfall auftrat, bei dem ein Entwickler eine scheinbar redundante Validierung entfernte und einen Datenfeed zu einem externen Laborsystem brach. Ihr erstes vierteljährliches Review deckte 34 protokollierte Einträge plus 19 aus dem Gedächtnis rekonstruierte ab. Elf kompensierten Probleme, die nicht mehr existierten: vier bezogen sich auf einen zwei Jahre zuvor ersetzten Zahlungsanbieter, drei umgingen eine Datenbankversion, die seitdem aufgerüstet worden war, und einer war für einen Kunden eingeführt worden, der kein Kunde mehr war. Das Entfernen dieser elf brauchte neun Tage und beseitigte zwei wiederkehrende monatliche manuelle Schritte. Der teuerste einzelne Eintrag stellte sich als organisatorisch heraus — ein Finanzsachbearbeiter, der ungefähr zwei Tage pro Monat mit dem Abgleich von Datensätzen verbrachte, weil zwei Subsysteme sich nicht darüber einig waren, wie sie monatsmittige Planänderungen handhaben sollten, ein Workaround, der seit sechs Jahren bestand und nie in irgendeiner Technologiediskussion aufgetaucht war.

Das Trendmaß änderte, wie über die Degradation des Teams gesprochen wurde. Über achtzehn Monate wuchs das Register von 34 auf 61 Einträge, wobei sich das Wachstum in einem Subsystem konzentrierte und sich nach einer Periode von Fristendruck beschleunigte. Ihrem Direktor als Diagramm präsentiert — pro Quartal hinzugefügte Workarounds, neben der entfernten Anzahl — machte es ein Argument, das die vorherigen Beschreibungen des Teams über sich ansammelnde technische Schulden nicht gemacht hatten. Die Antwort war kein Modernisierungsprogramm, sondern etwas Nützlicheres: eine ständige Regel, dass jeder unter Fristendruck eingeführte Workaround ein verpflichtendes Entfernungsreview bei der nächsten Quartalsüberprüfung nach sich zog, was die Nettowachstumsrate im folgenden Jahr halbierte.
