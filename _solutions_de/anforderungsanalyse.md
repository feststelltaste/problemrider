---
title: Anforderungsanalyse
description: Systematische Erhebung, Analyse und Dokumentation funktionaler
  Anforderungen.
category:
- Requirements
- Process
- Communication
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/requirements-analysis/
problems:
- inadequate-requirements-gathering
- requirements-ambiguity
- implementation-starts-without-design
- poor-planning
- planning-dysfunction
- scope-creep
- feature-creep
- feature-bloat
- large-feature-scope
- unrealistic-deadlines
- unrealistic-schedule
- stakeholder-developer-communication-gap
- no-continuous-feedback-loop
- frequent-changes-to-requirements
layout: solution
lang: de
en_slug: requirements-analysis
related_solutions:
- slug: evolutionary-requirements-development
  similarity: 0.85
- slug: user-stories
  similarity: 0.8
- slug: risk-analysis
  similarity: 0.8
- slug: requirements-traceability-matrix
  similarity: 0.75
- slug: functional-gap-analysis
  similarity: 0.75
- slug: on-site-customer
  similarity: 0.75
---

## Description

Anforderungsanalyse erhebt und dokumentiert systematisch, was ein System tatsächlich tun muss, unter Nutzung strukturierter Techniken — kontextuelle Befragung, Reverse-Engineering des tatsächlichen Verhaltens des bestehenden Systems, Zerlegung vager Ziele in testbare Abnahmekriterien — statt eines einzelnen Workshops oder der besten Vermutung eines Stakeholders. In der Legacy-Modernisierung ist dies besonders folgenreich, weil die echte Spezifikation oft in jahrzehntealtem Code und den gewohnten Workarounds langjähriger Mitarbeiter eingeschlossen ist, die nicht mehr daran denken, sie zu erwähnen, was bedeutet, dass Anforderungen, die nur durch die Frage „was wollen Sie" erhoben werden, verlässlich die Einschränkungen übersehen, die ein Ersatzsystem später tatsächlich brechen. Diese Zeit im Voraus zu investieren erfasst die Lücke zwischen dokumentiertem und tatsächlichem Systemverhalten, bevor sie zu teurem Nacharbeitsaufwand wird, obwohl derselbe Instinkt, zu weit getrieben, Analyselähmung produziert — ein echtes Risiko in Legacy-Kontexten, wo der vollständige Umfang oft echt im Voraus unbekannt ist, sodass die Analyse auf das begrenzt werden muss, was das aktuelle Projekt braucht, statt auf eine vollständige Bestandsaufnahme des gesamten Systems.

## How to Apply ◆

> In Legacy-Systemen, in denen Anforderungen oft in den Köpfen langjähriger Mitarbeiter eingeschlossen, in jahrzehntealtem Code eingebettet oder in veralteten Spezifikationen dokumentiert sind, die die Realität nicht mehr widerspiegeln, ersetzt systematische Anforderungsanalyse Rätselraten durch strukturierte Entdeckung.

- Beginnen Sie jedes Projekt oder größere Feature mit einer dedizierten Anforderungsanalysephase, selbst wenn sie kurz ist. Die Analyse muss keine umfassende Spezifikation produzieren; sie muss genug Klarheit produzieren, damit das Team das erste Inkrement mit Vertrauen beginnen kann. In Legacy-Kontexten bedeutet das, zu identifizieren, was das aktuelle System tut (Verhaltensanforderungen), was es weiterhin tun sollte (Bewahrungsanforderungen), und was sich ändern muss (Verbesserungsanforderungen).
- Führen Sie strukturierte Stakeholder-Interviews durch, die über „was wollen Sie?" hinausgehen. Nutzen Sie Techniken wie kontextuelle Befragung — Beobachtung von Nutzern bei ihrer tatsächlichen Arbeit mit dem Legacy-System —, um Anforderungen aufzudecken, die Stakeholder nicht artikulieren können, weil sie gewohnheitsmäßig geworden sind. Legacy-System-Nutzer entwickeln oft komplexe Workarounds, die kritische Geschäftslogik kodieren, die in keiner Dokumentation erfasst ist.
- Analysieren Sie das bestehende Legacy-System als Anforderungsquelle: untersuchen Sie Bildschirmabläufe, Datenbankschemata, im Code kodierte Geschäftsregeln, Batch-Job-Zeitpläne und Integrationsschnittstellen. Dieses Reverse-Engineering deckt implizite Anforderungen auf, die zu Fehlschlägen führen, wenn sie übersehen werden. Dokumentieren Sie, was das System tatsächlich tut, nicht was die Dokumentation behauptet, weil sich die beiden in langlebigen Systemen häufig unterscheiden.
- Zerlegen Sie Anforderungen in testbare Abnahmekriterien, bevor die Entwicklung beginnt. Ersetzen Sie vage Anforderungen wie „das System sollte schnell sein" durch spezifische, messbare Kriterien wie „Suchergebnisse müssen innerhalb von 500 Millisekunden für Abfragen gegen bis zu 100.000 Datensätze zurückgegeben werden." Diese Zerlegung ist es, was Anforderungsmehrdeutigkeit davon abhält, zu Implementierungs-Nacharbeit zu werden.
- Identifizieren und dokumentieren Sie Einschränkungen explizit: regulatorische Anforderungen, Integrationsabhängigkeiten, Datenvolumenerwartungen und nichtfunktionale Anforderungen wie Performance und Verfügbarkeit. Legacy-Modernisierungsprojekte scheitern häufig, weil diese Einschränkungen angenommen statt analysiert werden, und das neue System sie nicht erfüllen kann.
- Bilden Sie Abhängigkeiten zwischen Anforderungen ab, um zu identifizieren, welche zusammen implementiert werden müssen und welche unabhängig geliefert werden können. Dieses Abhängigkeits-Mapping ist essenziell, um große Feature-Umfänge in ausliefer­bare Inkremente aufzuteilen und für realistische Terminplanung, die Reihenfolgebeschränkungen berücksichtigt.
- Validieren Sie Anforderungen mit Stakeholdern durch konkrete Beispiele, Prototypen oder Durchgänge, bevor Sie sich auf die Implementierung festlegen. In Legacy-Kontexten sind Seite-an-Seite-Vergleiche des bestehenden Verhaltens und des vorgeschlagenen Verhaltens besonders effektiv, weil sie Stakeholdern einen Bezugspunkt geben, um zu bewerten, ob das neue System ihre Bedürfnisse erfüllen wird.
- Pflegen Sie ein Anforderungs-Rückverfolgungsprotokoll, das jede Anforderung mit ihrer Quelle (Stakeholder, Regulierung, bestehendes Systemverhalten), ihrem Implementierungsstatus und ihrer Verifikationsmethode verbindet. Diese Rückverfolgbarkeit ist es, was verhindert, dass Anforderungen zwischen Analyse und Implementierung verloren gehen, und liefert die Grundlage für genaue Planung.

## Tradeoffs ⇄

> Systematische Anforderungsanalyse investiert Zeit im Voraus, um die weit höheren Kosten des Bauens des Falschen zu reduzieren, muss aber kalibriert werden, um das gegenteilige Extrem der Analyselähmung zu vermeiden, die Auslieferung unbegrenzt verzögert.

**Vorteile:**

- Verhindert das Muster, dass Implementierung ohne Design beginnt, indem ein klares Verständnis dessen etabliert wird, was gebaut werden muss, bevor Codierung beginnt, und reduziert den strukturellen Nacharbeitsaufwand, der aus der Entdeckung grundlegender Anforderungen mitten in der Implementierung resultiert.
- Verbessert die Schätzgenauigkeit dramatisch, indem tatsächlicher Umfang, Komplexität und Abhängigkeiten offengelegt werden, bevor Verpflichtungen eingegangen werden, und adressiert direkt unrealistische Fristen und Zeitpläne, die aus Plänen resultieren, die auf unvollständigem Verständnis basieren.
- Reduziert Umfangsausweitung, indem eine klare, vereinbarte Basislinie von Anforderungen etabliert wird, gegen die vorgeschlagene Ergänzungen bewertet werden können, was die Kosten der Umfangserweiterung sichtbar und bewusst macht.
- Legt das Risiko von Feature-Aufblähung früh offen: Wenn die Anforderungsanalyse zeigt, dass ein vorgeschlagener Feature-Umfang größer ist, als das Team innerhalb der Beschränkungen liefern kann, kann der Umfang vor der Entwicklungsinvestition statt danach verhandelt werden.
- Schafft ein gemeinsames Verständnis zwischen Stakeholdern und Entwicklern, indem Geschäftsbedürfnisse in spezifische, testbare Kriterien übersetzt werden, und schließt die Kommunikationslücke, die fehlangepasste Ergebnisse verursacht.
- Identifiziert Anforderungskonflikte und -abhängigkeiten früh genug, um sie durch Verhandlung zu lösen, statt sie während der Implementierung zu entdecken, wenn die Lösung weit teurer ist.

**Kosten und Risiken:**

- Anforderungsanalyse, die versucht, Vollständigkeit zu erreichen, bevor irgendeine Entwicklung beginnt, kann Analyselähmung erzeugen und Auslieferung verzögern, während das Team einem unerreichbaren Ziel perfekter Spezifikation nachjagt — dies ist besonders gefährlich in Legacy-Kontexten, wo der vollständige Umfang im Voraus echt unbekannt ist.
- In sich schnell ändernden Geschäftsumgebungen könnten während der Analyse erfasste Anforderungen veraltet werden, bevor sie implementiert werden, was erfordert, dass der Analyseprozess iterativ statt einmalig ist.
- Legacy-Systeme mit Jahrzehnten angesammelten Verhaltens könnten einen überwältigenden Analyseumfang darstellen; das Team muss diszipliniert sein, nur das zu analysieren, was für das aktuelle Projekt relevant ist, statt ein vollständiges Reverse-Engineering des gesamten Systems zu versuchen.
- Anforderungsanalyse erfordert Zugang zu sachkundigen Stakeholdern und Fachexperten, die oft dieselben Personen sind, die das Legacy-System warten und zu beschäftigt sind, um es zu analysieren — die Planung ihrer Zeit für Analysesitzungen konkurriert mit betrieblichen Anforderungen.
- Formale Anforderungsdokumentation kann ein falsches Gefühl von Vollständigkeit erzeugen, das die laufende Verfeinerung entmutigt, die gebraucht wird, während sich das Verständnis während der Entwicklung vertieft.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie systematische Anforderungsanalyse häufige Fehlermuster in der Legacy-System-Modernisierung verhindert.

Ein regionales Krankenhaus ersetzte sein Patientenaufnahmesystem, das zweiundzwanzig Jahre in Produktion gewesen war. Der erste Modernisierungsversuch war gescheitert, nachdem neun Monate lang gegen Anforderungen gebaut worden war, die in einem einzigen Workshop mit Abteilungsleitern erhoben wurden, nur um während des Nutzertests zu entdecken, dass das System die komplexen Patientenverlegungsarbeitsabläufe nicht handhaben konnte, auf die sich Frontline-Personal täglich verließ. Für den zweiten Versuch verbrachte das Team vier Wochen mit strukturierter Anforderungsanalyse: Sie beobachteten Aufnahmepersonal während Spitzenzeiten, analysierten die Datenbank des Legacy-Systems, um tatsächliche Datenflüsse zu verstehen, interviewten Krankenschwestern und Ärzte über Grenzfälle und bildeten jeden Integrationspunkt mit Abrechnungs-, Apotheken- und Laborsystemen ab. Die Analyse offenbarte dreiundsechzig implizite Anforderungen, die kein Stakeholder in Interviews erwähnt hatte, weil sie so gewohnheitsmäßig waren, dass das Personal nicht daran dachte, sie zu artikulieren. Am kritischsten deckte die Analyse auf, dass die Patientenverlegungslogik des Legacy-Systems von einer spezifischen Sequenz von Datenbankaktualisierungen abhing, an die das Personal seinen Arbeitsablauf angepasst hatte — eine Einschränkung, die im neuen System bewahrt oder explizit neu gestaltet werden musste. Die vierwöchige Analyseinvestition verhinderte die neun Monate verschwendeter Entwicklung, die der erste Versuch verbraucht hatte.

Ein Finanzdienstleistungsunternehmen plante, sein Handelsabwicklungssystem mit einer anfänglichen Schätzung von zwölf Monaten und einem Budget von zwei Millionen Dollar zu modernisieren. Bevor Ressourcen gebunden wurden, führte der technische Architekt eine sechswöchige Anforderungsanalyse durch, die Reverse-Engineering der Abgleichslogik des Legacy-Systems, die Abbildung von Integrationsabhängigkeiten mit acht externen Gegenparteisystemen und die Dokumentation der regulatorischen Berichtsanforderungen umfasste. Die Analyse offenbarte, dass die Abgleichs-Engine des Legacy-Systems siebzehn undokumentierte Ausnahmebehandlungspfade enthielt, die über fünfzehn Jahre als Reaktion auf spezifische Gegenparteiausfälle hinzugefügt worden waren. Sie offenbarte außerdem, dass drei der acht Gegenpartei-Integrationen proprietäre Protokolle nutzten, die der vorgeschlagene neue Technologie-Stack nicht nativ unterstützte. Die realistische Schätzung basierend auf der Analyse betrug zwanzig Monate und drei Millionen Dollar. Obwohl die Analyse unwillkommene Neuigkeiten lieferte, verhinderte sie, dass sich die Organisation auf ein Budget und einen Zeitplan festlegte, die zu einem gescheiterten Projekt geführt hätten. Der leitende Sponsor nutzte die detaillierte Analyse, um angemessene Finanzierung zu sichern und realistische Erwartungen mit dem Vorstand zu setzen, und das Projekt lieferte letztlich in neunzehn Monaten — vor der revidierten Schätzung und deutlich unter dem revidierten Budget, weil die Analyse die kostspieligen Überraschungen beseitigt hatte, die frühere Modernisierungsversuche der Organisation entgleist hatten.
