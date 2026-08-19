---
title: Evolutionäre Anforderungsentwicklung
description: Schrittweise Detaillierung und Verfeinerung von Anforderungen über den
  gesamten Projektverlauf.
category:
- Requirements
- Process
- Communication
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/evolutionary-requirements-development/
problems:
- inadequate-requirements-gathering
- requirements-ambiguity
- frequent-changes-to-requirements
- implementation-starts-without-design
- scope-creep
- changing-project-scope
- stakeholder-developer-communication-gap
- no-continuous-feedback-loop
- eager-to-please-stakeholders
- missed-deadlines
- planning-dysfunction
- stakeholder-frustration
- feature-creep
layout: solution
lang: de
en_slug: evolutionary-requirements-development
related_solutions:
- slug: requirements-analysis
  similarity: 0.85
- slug: specification-by-example
  similarity: 0.75
- slug: iterative-development
  similarity: 0.75
- slug: on-site-customer
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: behavior-driven-development-bdd
  similarity: 0.75
---

## Description

Evolutionäre Anforderungsentwicklung verfeinert Anforderungen schrittweise, in kurzen Zyklen mit konkreten Beispielen, statt zu versuchen, alles vollständig zu spezifizieren, bevor die Implementierung beginnt. Dies ist besonders wichtig für Legacy-Modernisierung, wo die ursprünglichen Stakeholder, die eine vollständige Spezifikation hätten schreiben können, häufig gegangen sind und die tatsächliche Anforderung das ist, was auch immer undokumentiertes Verhalten das aktuelle System zufällig zeigt — etwas, das nur durch iterative Verfeinerung mit den Menschen entdeckt werden kann, die das System heute nutzen, nicht aus einem Dokument abgeleitet. Einen kontinuierlich gepflegten Backlog statt einer einmal eingefrorenen und genehmigten Spezifikation zu pflegen hält den Prozess ehrlich darüber, wie viel zu einem gegebenen Zeitpunkt genuin bekannt ist, obgleich es anhaltende Stakeholder-Verfügbarkeit erfordert, die oft knapp ist, gerade weil dieselben Menschen damit beschäftigt sind, das Legacy-System am Laufen zu halten.

## How to Apply ◆

> In Legacy-Systemen, wo Anforderungen oft vor Jahren definiert wurden und die ursprünglichen Stakeholder möglicherweise nicht mehr verfügbar sind, ersetzt evolutionäre Anforderungsentwicklung das unmögliche Ziel vollständiger vorgelagerter Spezifikation durch einen disziplinierten Prozess schrittweiser Verfeinerung, der mit dem tatsächlichen Verständnis Schritt hält.

- Beginnen Sie jedes Projektinkrement mit einem Anforderungs-Workshop, der Entwickler, Geschäfts-Stakeholder und Nutzer zusammenbringt, um gemeinsam die nächste Anforderungsscheibe zu verfeinern. In Legacy-Kontexten dienen diese Workshops doppeltem Zweck: Sie erfassen aktuelle Bedürfnisse und fördern undokumentierte Verhaltensweisen des bestehenden Systems zutage, die bewahrt oder bewusst geändert werden müssen.
- Nutzen Sie leichtgewichtige Spezifikationsformate — User Stories mit Abnahmekriterien, Specification by Example oder Entscheidungstabellen — statt schwergewichtiger Anforderungsdokumente, die veralten, bevor die Entwicklung beginnt. Bei Legacy-Modernisierung sollten Abnahmekriterien explizit angeben, ob bestehendes Verhalten bewahrt oder bewusst geändert wird.
- Pflegen Sie einen lebenden Anforderungs-Backlog, der kontinuierlich gepflegt wird, statt einer Anforderungsspezifikation, die einmal genehmigt und dann als unveränderlich behandelt wird. Ordnen Sie Posten nach Geschäftswert und technischem Risiko und akzeptieren Sie, dass niedriger eingestufte Posten sich erheblich ändern könnten, bis das Team sie erreicht.
- Verlangen Sie, dass jede Anforderung mindestens ein konkretes Beispiel erwarteten Verhaltens enthält, ausgedrückt in Begriffen, die sowohl Entwickler als auch Stakeholder verifizieren können. Abstrakte Anforderungen wie „das System sollte schnell sein" oder „die Nutzererfahrung verbessern" sind nicht handlungsfähig und sollten abgelehnt werden, bis sie als testbare Szenarien ausgedrückt werden können.
- Führen Sie regelmäßige Backlog-Refinement-Sitzungen durch, in denen das Team bevorstehende Anforderungen überprüft, Mehrdeutigkeiten identifiziert, klärende Fragen stellt und Aufwand schätzt. Diese Sitzungen sollten mindestens eine Iteration vor der Entwicklung stattfinden, um zu verhindern, dass das Team mit unklaren Anforderungen mit der Implementierung beginnt.
- Etablieren Sie explizite „gerade genug"-Design-Checkpoints vor Beginn der Implementierung: Das Team bespricht architektonische Implikationen und Komponenteninteraktionen für die anstehenden Anforderungen, ohne umfassende Design-Dokumente zu erzeugen. Für Legacy-Systeme beinhaltet dies die Analyse, wie neue Anforderungen mit bestehenden Systemeinschränkungen interagieren.
- Erstellen Sie ein gemeinsames Glossar von Domänenbegriffen, auf das sich sowohl geschäftliche als auch technische Stakeholder einigen, und pflegen Sie es als lebendes Dokument. In Legacy-Systemen bedeutet derselbe Begriff oft unterschiedlichen Abteilungen unterschiedliche Dinge, weil sich das System über Jahrzehnte unter unterschiedlichen Teams entwickelt hat.
- Verfolgen Sie Anforderungsänderungen als normalen Teil des Prozesses statt sie als Versagen zu behandeln. Messen Sie die Änderungsrate, um zu identifizieren, wann Instabilität ein tieferes Problem anzeigt — wie unklare Produktvision oder widersprüchliche Stakeholder-Interessen — statt gesunder Evolution.

## Tradeoffs ⇄

> Evolutionäre Anforderungsentwicklung tauscht den Komfort einer „vollständigen" Spezifikation gegen die Fähigkeit, sich anzupassen, während das Verständnis wächst, was besonders wertvoll in Legacy-Kontexten ist, wo die wahren Anforderungen oft entdeckt statt spezifiziert werden.

**Vorteile:**

- Verhindert das kostspielige Muster, gegen eine Spezifikation zu bauen, die sich als falsch herausstellt, was besonders gefährlich in Legacy-Modernisierung ist, wo die Lücke zwischen dokumentierten Anforderungen und tatsächlichem Systemverhalten enorm sein kann.
- Reduziert Anforderungsmehrdeutigkeit, indem detaillierte Spezifikation aufgeschoben wird, bis das Team genug Kontext hat, um präzise, testbare Kriterien zu schreiben, statt Monate vor der Implementierung Details zu raten.
- Passt sich natürlich Anforderungsänderungen an, indem sie als erwartet statt störend behandelt werden, was die gegnerische Dynamik zwischen Stakeholdern, die Änderungen brauchen, und Entwicklern, die sich dagegen sträuben, reduziert.
- Schafft regelmäßige Gelegenheiten für Stakeholder und Entwickler, gemeinsames Verständnis aufzubauen, was die Kommunikationslücke schrittweise schließt, die viele Legacy-Projekte plagt.
- Ermöglicht frühere Erkennung widersprüchlicher Anforderungen, indem sie in konkreten Begriffen besprochen werden, bevor erheblicher Entwicklungsaufwand investiert wird.

**Kosten und Risiken:**

- Erfordert anhaltende Stakeholder-Verfügbarkeit während des gesamten Projekts, was schwierig sein kann, wenn Geschäftsexperten bereits überlastet sind, das Legacy-System zu unterstützen, das sie auch zu ersetzen versuchen.
- Teams, die daran gewöhnt sind, vollständige Spezifikationen zu erhalten, könnten sich unwohl fühlen, mit explizit unvollständigen Anforderungen zu arbeiten, und es als schlechte Planung statt als bewussten Inkrementalismus wahrnehmen.
- Ohne Disziplin können evolutionäre Anforderungen zu „wir werden es unterwegs herausfinden" entarten, was keine progressive Verfeinerung, sondern Abwesenheit von Planung ist — das Team muss eine klare Vision des Gesamtumfangs bewahren, selbst während sich Details weiterentwickeln.
- Stakeholder, die vorhersehbare langfristige Roadmaps erwarten, könnten mit einem Prozess kämpfen, der detaillierte Anforderungen bewusst bis näher an die Implementierung aufschiebt, was Planungsglaubwürdigkeitsbedenken auf organisatorischer Ebene schafft.
- In Festpreis- oder regulierten Kontexten kollidieren evolutionäre Anforderungen mit vertraglichen oder Compliance-Erwartungen an vorgelagerte Spezifikation, was sorgfältige Verhandlung darüber erfordert, wie Anforderungs-Baselines verwaltet werden.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie evolutionäre Anforderungsentwicklung die Herausforderungen unvollständiger oder sich ändernder Anforderungen in Legacy-System-Kontexten adressiert.

Ein Logistikunternehmen modernisierte sein zwölf Jahre in Produktion befindliches Auftragsverwaltungssystem. Die ursprünglichen Anforderungsdokumente waren veraltet und unvollständig, und die drei Business-Analysten, die das System verstanden, waren in Rente gegangen. Statt zu versuchen, eine vollständige Spezifikation per Reverse Engineering zu erschließen, übernahm das Team zweiwöchige Verfeinerungszyklen, in denen sie mit Lagerbetreibern und Kundenservice-Personal zusammenarbeiteten, um die Anforderungen für die nächsten zwei Features im Detail mittels konkreter Beispiele aus echten Bestellungen zu dokumentieren. Jeder Zyklus produzierte testbare Abnahmekriterien und förderte undokumentierte, im Legacy-Code eingebettete Geschäftsregeln zutage. Über sechs Monate baute das Team eine zuverlässige, wachsende Spezifikation, die immer korrekt war, weil sie immer aktuell war, und sie vermieden die dreimonatige vorgelagerte Analysephase, die zwei vorherige Modernisierungsversuche entgleist hatte.

Ein Versicherungsunternehmen begann einen Ersatz der Policenverwaltung, bei dem Stakeholder anfangs ein 200-seitiges Anforderungsdokument bereitstellten. Das Entwicklungsteam entdeckte innerhalb des ersten Monats, dass viele Anforderungen dem tatsächlichen Verhalten des Legacy-Systems widersprachen, und Geschäftsnutzer bestätigten, dass die dokumentierten Regeln vor Jahren durch manuelle Prozesse überschrieben worden waren. Das Team wechselte zu evolutionärer Anforderungsentwicklung, arbeitete mit Schadensregulierern zusammen, um Anforderungen einen Geschäftsprozess nach dem anderen zu definieren, und validierte jeden gegen das tatsächliche Verhalten des Legacy-Systems, bevor gebaut wurde. Dieser Ansatz verlängerte den anfänglichen Zeitplan um drei Wochen, eliminierte aber geschätzte vier Monate Nacharbeit, die aus dem Bauen gegen die ungenaue Spezifikation resultiert hätten. Wichtiger noch gab es den Geschäfts-Stakeholdern Vertrauen, dass der Ersatz tatsächlich ihre echten Arbeitsabläufe handhaben würde, nicht nur die offiziell dokumentierten.
